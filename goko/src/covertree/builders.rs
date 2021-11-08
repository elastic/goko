/*
* Licensed to Elasticsearch B.V. under one or more contributor
* license agreements. See the NOTICE file distributed with
* this work for additional information regarding copyright
* ownership. Elasticsearch B.V. licenses this file to you under
* the Apache License, Version 2.0 (the "License"); you may
* not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*  http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/
use super::data_caches::*;
use super::layer::*;
use super::node::*;
use super::*;
use crate::plugins::TreePluginSet;
use crate::*;
use pbr::ProgressBar;
use rand::rngs::SmallRng;
use rand::SeedableRng;
use std::cmp::{max, min};
use std::fs::read_to_string;
use std::path::Path;
use std::sync::{atomic, Arc, RwLock};
use yaml_rust::YamlLoader;

use crossbeam_channel::{unbounded, Receiver, Sender};
use errors::GokoResult;

use std::time::Instant;

#[derive(Debug)]
struct BuilderNode {
    parent_address: Option<NodeAddress>,
    scale_index: i32,
    covered: CoveredData,
}

type NodeSplitResult<D> = GokoResult<(NodeAddress, CoverNode<D>)>;

impl BuilderNode {
    fn new<D: PointCloud>(
        parameters: &CoverTreeParameters<D>,
        partition_type: PartitionType,
    ) -> GokoResult<BuilderNode> {
        let covered = match partition_type {
            PartitionType::Nearest => CoveredData::NearestCoveredData(
                NearestCoveredData::new::<D>(&parameters.point_cloud)?,
            ),
            PartitionType::First => {
                CoveredData::FirstCoveredData(FirstCoveredData::new::<D>(&parameters.point_cloud)?)
            }
        };
        let scale_index = (covered.max_distance()).log(parameters.scale_base).ceil() as i32;
        assert!(
            -64 <= scale_index && scale_index < 447,
            "Scale index needs to be in [-64, 447], it's {}. Increase the scale base.",
            scale_index
        );
        Ok(BuilderNode {
            parent_address: None,
            scale_index,
            covered,
        })
    }

    #[inline]
    fn address(&self) -> NodeAddress {
        (self.scale_index, self.covered.center_index()).into()
    }

    fn split_parallel<D: PointCloud>(
        self,
        parameters: &Arc<CoverTreeParameters<D>>,
        node_sender: &Arc<Sender<NodeSplitResult<D>>>,
    ) {
        let parameters = Arc::clone(parameters);
        let node_sender = Arc::clone(node_sender);
        rayon::spawn(move || {
            let na = self.address();
            match self.split(&parameters) {
                Ok((new_node, mut new_nodes)) => {
                    node_sender.send(Ok((na, new_node))).unwrap();
                    while let Some(node) = new_nodes.pop() {
                        node.split_parallel(&parameters, &node_sender);
                    }
                }
                Err(e) => node_sender.send(Err(e)).unwrap(),
            };
        });
    }

    fn split<D: PointCloud>(
        self,
        parameters: &Arc<CoverTreeParameters<D>>,
    ) -> GokoResult<(CoverNode<D>, Vec<BuilderNode>)> {
        let scale_index = self.scale_index;
        let current_address =
            unsafe { NodeAddress::new_unchecked(scale_index, self.covered.center_index()) };
        let mut node = CoverNode::new(self.parent_address, current_address);
        let radius = self.covered.max_distance();
        node.set_radius(radius);
        /* Occasionally there's a small cluster split off of at a low min_res_index.
        This brings the scale-index down/min_res_index up quickly, locally.
        */
        let mut new_nodes = if self.covered.len() <= parameters.leaf_cutoff
            || scale_index < parameters.min_res_index
        {
            //println!("== This is getting cut down by parameters ==");
            node.insert_singletons(self.covered.into_indexes());
            vec![]
        } else {
            let next_scale_index = min(
                scale_index - 1,
                max(
                    radius.log(parameters.scale_base).ceil() as i32,
                    parameters.min_res_index,
                ),
            );
            match self.covered {
                CoveredData::FirstCoveredData(covered) => BuilderNode::split_first(
                    &mut node,
                    current_address,
                    covered,
                    next_scale_index,
                    parameters,
                )?,
                CoveredData::NearestCoveredData(covered) => BuilderNode::split_nearest(
                    &mut node,
                    current_address,
                    covered,
                    next_scale_index,
                    parameters,
                )?,
            }
        };

        if new_nodes.len() == 1 && new_nodes[0].covered.len() == 1 {
            node.remove_children();
            parameters
                .total_nodes
                .fetch_sub(1, atomic::Ordering::SeqCst);
            node.insert_singletons(new_nodes.pop().unwrap().covered.into_indexes());
        }

        // This node is done, send it in
        //println!("=====================");
        Ok((node, new_nodes))
    }

    fn split_nearest<D: PointCloud>(
        parent_node: &mut CoverNode<D>,
        parent_address: NodeAddress,
        covered: NearestCoveredData,
        split_scale_index: i32,
        parameters: &Arc<CoverTreeParameters<D>>,
    ) -> GokoResult<Vec<BuilderNode>> {
        let mut small_rng: SmallRng = match parameters.rng_seed {
            Some(seed) => SmallRng::seed_from_u64(seed ^ parent_address.point_index() as u64),
            None => SmallRng::from_entropy(),
        };
        let next_scale = parameters.scale_base.powi(split_scale_index);
        let (nested_potential, mut splits) =
            covered.split(next_scale, &parameters.point_cloud, &mut small_rng)?;
        let mut new_nodes = Vec::new();

        let mut inserts = Vec::new();

        for potential in splits.drain(0..) {
            if potential.len() == 1 && parameters.use_singletons {
                parent_node.insert_singleton(potential.center_index);
            } else {
                inserts.push(((split_scale_index, potential.center_index), potential.len()));

                let new_node = BuilderNode {
                    parent_address: Some(parent_address),
                    scale_index: split_scale_index,
                    covered: CoveredData::NearestCoveredData(potential),
                };
                new_nodes.push(new_node);
                parameters
                    .total_nodes
                    .fetch_add(1, atomic::Ordering::SeqCst);
            }
        }
        if !inserts.is_empty() || !(nested_potential.len() == 1 && parameters.use_singletons) {
            parent_node.insert_nested_child(split_scale_index, nested_potential.len())?;

            let new_node = BuilderNode {
                parent_address: Some(parent_address),
                scale_index: split_scale_index,
                covered: CoveredData::NearestCoveredData(nested_potential),
            };
            new_nodes.push(new_node);
            parameters
                .total_nodes
                .fetch_add(1, atomic::Ordering::SeqCst);

            for ((split_scale_index, potential_center_index), potential_len) in inserts {
                parent_node.insert_child(
                    unsafe {
                        NodeAddress::new_unchecked(split_scale_index, potential_center_index)
                    },
                    potential_len,
                )?;
            }
        }

        Ok(new_nodes)
    }

    fn split_first<D: PointCloud>(
        parent_node: &mut CoverNode<D>,
        parent_address: NodeAddress,
        covered: FirstCoveredData,
        split_scale_index: i32,
        parameters: &Arc<CoverTreeParameters<D>>,
    ) -> GokoResult<Vec<BuilderNode>> {
        let mut small_rng: SmallRng = match parameters.rng_seed {
            Some(seed) => SmallRng::seed_from_u64(seed ^ parent_address.raw()),
            None => SmallRng::from_entropy(),
        };
        let mut new_nodes = Vec::new();

        let next_scale = parameters.scale_base.powi(split_scale_index);

        /*
        We get a bunch of points (close) that are within `next_scale` of the center.
        we also get a bunch of points further out (new_fars). For these we need to find centers.
        */

        let (close, mut fars) = covered.split(next_scale).unwrap();
        //println!("== Split loop setup with scale {}, and scale index {} ==", next_scale, next_scale_index);
        //println!("\tCovered: {:?}", close);
        //println!("\tNot Covered: {:?}", fars);

        parent_node.insert_nested_child(split_scale_index, close.len())?;
        let new_node = BuilderNode {
            parent_address: Some(parent_address),
            scale_index: split_scale_index,
            covered: CoveredData::FirstCoveredData(close),
        };
        new_nodes.push(new_node);
        parameters
            .total_nodes
            .fetch_add(1, atomic::Ordering::SeqCst);
        /*
        First we make the covered child. This child has the same center as it's parent and it
        covers the points that are in the "close" set.
        */

        /*
        We have the core loop that makes new points. We check that the new_fars' exist (split
        returns None if there arn't any points more than next_scale from the center), then if
        it does we split it again.

        The DistCache is responsible for picking new centers each time there's a split (to
        ensure it always returns a valid DistCache).
        */

        while fars.len() > 0 {
            let new_close =
                fars.pick_center(next_scale, &parameters.point_cloud, &mut small_rng)?;
            //println!("\t\t [{}] New Covered: {:?}",split_count, new_close);
            if new_close.len() == 1 && parameters.use_singletons {
                /*
                We have a vast quantity of internal ourliers. These are singleton points that are
                at least next_scale away from each other. These could be fully fledged leaf nodes,
                or we can short circut them and just store a reference.

                On malware data 80% of the data are outliers of this type. References are a significant
                ram savings.
                */
                parent_node.insert_singleton(new_close.center_index);
            } else {
                parent_node.insert_child(
                    unsafe {
                        NodeAddress::new_unchecked(split_scale_index, new_close.center_index)
                    },
                    new_close.len(),
                )?;
                let new_node = BuilderNode {
                    parent_address: Some(parent_address),
                    scale_index: split_scale_index,
                    covered: CoveredData::FirstCoveredData(new_close),
                };
                new_nodes.push(new_node);
                parameters
                    .total_nodes
                    .fetch_add(1, atomic::Ordering::SeqCst);
            }
        }

        Ok(new_nodes)
    }
}

/// A construction object for a covertree. See [`crate::covertree::CoverTreeParameters`] for docs
#[derive(Debug)]
pub struct CoverTreeBuilder {
    pub(crate) scale_base: f32,
    pub(crate) leaf_cutoff: usize,
    pub(crate) min_res_index: i32,
    pub(crate) use_singletons: bool,
    pub(crate) partition_type: PartitionType,
    pub(crate) verbosity: u32,
    pub(crate) rng_seed: Option<u64>,
}

impl Default for CoverTreeBuilder {
    fn default() -> CoverTreeBuilder {
        CoverTreeBuilder {
            scale_base: 2.0,
            leaf_cutoff: 1,
            min_res_index: -10,
            use_singletons: true,
            partition_type: PartitionType::Nearest,
            verbosity: 0,
            rng_seed: None,
        }
    }
}

impl CoverTreeBuilder {
    /// Creates a new builder with sensible defaults.
    pub fn new() -> CoverTreeBuilder {
        CoverTreeBuilder {
            scale_base: 2.0,
            leaf_cutoff: 1,
            min_res_index: -10,
            use_singletons: true,
            partition_type: PartitionType::Nearest,
            verbosity: 0,
            rng_seed: None,
        }
    }

    /// Creates a builder from an open yaml object
    pub fn from_yaml<P: AsRef<Path>>(path: P) -> Self {
        let config = read_to_string(&path).expect("Unable to read config file");
        let params_files = YamlLoader::load_from_str(&config).unwrap();
        let params = &params_files[0];
        let partition_type = if "first" == params["partition_type"].as_str().unwrap_or("nearest") {
            PartitionType::First
        } else {
            PartitionType::Nearest
        };
        CoverTreeBuilder {
            scale_base: params["scale_base"].as_f64().unwrap_or(2.0) as f32,
            leaf_cutoff: params["leaf_cutoff"].as_i64().unwrap_or(1) as usize,
            min_res_index: params["min_res_index"].as_i64().unwrap_or(-10) as i32,
            use_singletons: params["use_singletons"].as_bool().unwrap_or(true),
            partition_type,
            verbosity: params["verbosity"].as_i64().unwrap_or(0) as u32,
            rng_seed: params["rng_seed"].as_i64().map(|i| i as u64),
        }
    }

    /// See [`crate::covertree::CoverTreeParameters`] for docs
    pub fn set_scale_base(&mut self, x: f32) -> &mut Self {
        self.scale_base = x;
        self
    }
    /// See [`crate::covertree::CoverTreeParameters`] for docs
    pub fn set_leaf_cutoff(&mut self, x: usize) -> &mut Self {
        self.leaf_cutoff = x;
        self
    }
    /// See [`crate::covertree::CoverTreeParameters`] for docs
    pub fn set_min_res_index(&mut self, x: i32) -> &mut Self {
        self.min_res_index = x;
        self
    }
    /// See [`crate::covertree::CoverTreeParameters`] for docs
    pub fn set_use_singletons(&mut self, x: bool) -> &mut Self {
        self.use_singletons = x;
        self
    }
    /// See [`crate::covertree::CoverTreeParameters`] for docs
    pub fn set_verbosity(&mut self, x: u32) -> &mut Self {
        self.verbosity = x;
        self
    }
    /// See [`crate::covertree::CoverTreeParameters`] for docs
    pub fn set_rng_seed(&mut self, x: u64) -> &mut Self {
        self.rng_seed = Some(x);
        self
    }
    /// Pass a point cloud object when ready.
    /// To do, make this point cloud an Arc
    pub fn build<D: PointCloud>(&self, point_cloud: Arc<D>) -> GokoResult<CoverTreeWriter<D>> {
        let parameters = CoverTreeParameters {
            total_nodes: atomic::AtomicUsize::new(1),
            scale_base: self.scale_base,
            leaf_cutoff: self.leaf_cutoff,
            min_res_index: self.min_res_index,
            use_singletons: self.use_singletons,
            partition_type: self.partition_type,
            point_cloud,
            verbosity: self.verbosity,
            rng_seed: self.rng_seed,
            plugins: RwLock::new(TreePluginSet::new()),
        };

        let root = BuilderNode::new(&parameters, self.partition_type)?;
        let root_address = root.address();
        let scale_range = root_address.scale_index() - parameters.min_res_index;
        let mut layers = Vec::with_capacity(scale_range as usize);
        layers.push(CoverLayerWriter::new(parameters.min_res_index - 1));
        for i in 0..(scale_range + 1) {
            layers.push(CoverLayerWriter::new(parameters.min_res_index + i as i32));
        }

        let (node_sender, node_receiver): (
            Sender<NodeSplitResult<D>>,
            Receiver<NodeSplitResult<D>>,
        ) = unbounded();

        let node_sender = Arc::new(node_sender);
        let parameters = Arc::new(parameters);
        root.split_parallel(&parameters, &node_sender);
        let mut pb = ProgressBar::new(1u64);
        if parameters.verbosity > 1 {
            pb.format("╢▌▌░╟");
        }

        let (_final_addresses_reader, final_addresses) = monomap::new();

        let mut cover_tree = CoverTreeWriter {
            parameters: Arc::clone(&parameters),
            layers,
            root_address,
            final_addresses,
        };

        let mut inserted_nodes: usize = 0;
        let now = Instant::now();
        loop {
            if let Ok(res) = node_receiver.recv() {
                let (new_addr, new_node) = res.unwrap();
                for singleton in new_node.singletons() {
                    cover_tree.final_addresses.insert(*singleton, new_addr);
                }
                if new_node.is_leaf() {
                    cover_tree
                        .final_addresses
                        .insert(new_addr.point_index(), new_addr);
                }
                unsafe {
                    cover_tree.insert_raw(new_addr, new_node);
                }
                inserted_nodes += 1;
                if parameters.verbosity > 1 {
                    pb.total = parameters.total_nodes.load(atomic::Ordering::SeqCst) as u64;
                    pb.inc();
                }
            }
            // Stop if there are enough done, and there are no more outstanding parameter references
            if inserted_nodes == parameters.total_nodes.load(atomic::Ordering::SeqCst) {
                break;
            }
        }
        if parameters.verbosity > 1 {
            println!("\nWriting layers...");
        }
        cover_tree.refresh();
        cover_tree.final_addresses.refresh();
        cover_tree.final_addresses.refresh();
        if parameters.verbosity > 1 {
            println!(
                "Finished building, took {:?} with {} per second",
                now.elapsed(),
                (inserted_nodes as f32) / now.elapsed().as_secs_f32()
            );
        }
        Ok(cover_tree)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::{thread, time};

    pub fn create_test_parameters(
        data: Vec<f32>,
        data_dim: usize,
    ) -> Arc<CoverTreeParameters<DefaultCloud<L2>>> {
        let point_cloud = Arc::new(DefaultCloud::<L2>::new(data, data_dim).unwrap());
        Arc::new(CoverTreeParameters {
            total_nodes: atomic::AtomicUsize::new(1),
            scale_base: 2.0,
            leaf_cutoff: 0,
            min_res_index: -9,
            use_singletons: true,
            partition_type: PartitionType::Nearest,
            point_cloud,
            verbosity: 0,
            rng_seed: Some(0),
            plugins: RwLock::new(TreePluginSet::new()),
        })
    }

    #[test]
    fn nearest_splits_conditions() {
        let mut data = Vec::with_capacity(20);
        for _i in 0..19 {
            data.push(rand::random::<f32>());
        }
        data.push(0.0);

        let test_parameters = create_test_parameters(data, 1);
        let build_node = BuilderNode::new(&test_parameters, PartitionType::Nearest).unwrap();
        let (scale_index, center_index) = (
            build_node.address().scale_index(),
            build_node.address().point_index(),
        );

        println!("{:?}", build_node);
        println!(
            "The center_index for the covered data should be 19 but is {}",
            build_node.covered.center_index()
        );
        assert!(center_index == 19);
        println!("The scale_index should be 0, but is {}", scale_index);
        assert!(scale_index == 0);

        let (new_node, unfinished_nodes) = build_node.split(&test_parameters).unwrap();
        println!("New Node: {:#?}", new_node);
        let split_count = test_parameters.total_nodes.load(atomic::Ordering::SeqCst) - 1;
        println!(
            "We should have split count be equal to the work count: split {} , work {}",
            split_count,
            unfinished_nodes.len()
        );
        println!("We shouldn't be a leaf: {}", new_node.is_leaf());
        assert!(!new_node.is_leaf());
        println!(
            "We should have children count be equal to the split count: {}",
            new_node.children_len()
        );
        assert!(new_node.children_len() == split_count);
    }

    #[test]
    fn first_splits_conditions() {
        let mut data = Vec::with_capacity(20);
        for _i in 0..19 {
            data.push(rand::random::<f32>());
        }
        data.push(0.0);

        let test_parameters = create_test_parameters(data, 1);
        let build_node = BuilderNode::new(&test_parameters, PartitionType::First).unwrap();
        let (scale_index, center_index) = (
            build_node.address().scale_index(),
            build_node.address().point_index(),
        );

        println!("{:?}", build_node);
        println!(
            "The center_index for the covered data should be 19 but is {}",
            build_node.covered.center_index()
        );
        assert!(center_index == 19);
        println!("The scale_index should be 0, but is {}", scale_index);
        assert!(scale_index == 0);

        let (new_node, unfinished_nodes) = build_node.split(&test_parameters).unwrap();
        let split_count = test_parameters.total_nodes.load(atomic::Ordering::SeqCst) - 1;
        println!(
            "We should have split count be equal to the work count: split {} , work {}",
            split_count,
            unfinished_nodes.len()
        );
        println!("We shouldn't be a leaf: {}", new_node.is_leaf());
        assert!(!new_node.is_leaf());
        println!(
            "We should have children count be equal to the split count: {}",
            new_node.children_len()
        );
        assert!(new_node.children_len() == split_count);
    }

    #[test]
    fn tree_first_structure_condition() {
        let data = vec![0.49, 0.491, -0.49, 0.0];
        let test_parameters = create_test_parameters(data, 1);

        let build_node = BuilderNode::new(&test_parameters, PartitionType::First).unwrap();

        let (node_sender, node_receiver): (
            Sender<GokoResult<(NodeAddress, CoverNode<DefaultCloud<L2>>)>>,
            Receiver<GokoResult<(NodeAddress, CoverNode<DefaultCloud<L2>>)>>,
        ) = unbounded();
        let node_sender = Arc::new(node_sender);

        build_node.split_parallel(&test_parameters, &node_sender);
        thread::sleep(time::Duration::from_millis(100));
        let split_count = test_parameters.total_nodes.load(atomic::Ordering::SeqCst) - 1;
        println!(
            "Split count {}, node_receiver {}",
            split_count,
            node_receiver.len()
        );
        assert!(split_count + 1 == node_receiver.len());
        assert!(split_count == 3);
        while let Ok(pat) = node_receiver.try_recv() {
            let (na, node) = pat.unwrap();
            println!("{:?}", node);
            match na.into() {
                Some((-1, 3)) => assert!(!node.is_leaf()),
                Some((-2, 3)) => assert!(node.is_leaf()),
                Some((-2, 2)) => assert!(node.is_leaf()),
                Some((-2, 0)) => assert!(!node.is_leaf()),
                Some((-2, 1)) => assert!(!node.is_leaf()),
                _ => {}
            };
        }
    }

    #[test]
    fn tree_nearest_structure_condition() {
        let data = vec![0.49, 0.491, -0.49, 0.0];
        let test_parameters = create_test_parameters(data, 1);

        let build_node = BuilderNode::new(&test_parameters, PartitionType::Nearest).unwrap();

        let (node_sender, node_receiver): (
            Sender<GokoResult<(NodeAddress, CoverNode<DefaultCloud<L2>>)>>,
            Receiver<GokoResult<(NodeAddress, CoverNode<DefaultCloud<L2>>)>>,
        ) = unbounded();
        let node_sender = Arc::new(node_sender);

        build_node.split_parallel(&test_parameters, &node_sender);
        thread::sleep(time::Duration::from_millis(100));
        let split_count = test_parameters.total_nodes.load(atomic::Ordering::SeqCst) - 1;
        println!(
            "Split count {}, node_receiver {}",
            split_count,
            node_receiver.len()
        );
        assert!(split_count + 1 == node_receiver.len());
        assert!(split_count == 3);
        while let Ok(pat) = node_receiver.try_recv() {
            let (na, node) = pat.unwrap();
            println!("{:?}", node);

            match na.into() {
                Some((-1, 3)) => assert!(!node.is_leaf()),
                Some((-2, 3)) => assert!(node.is_leaf()),
                Some((-2, 2)) => assert!(node.is_leaf()),
                Some((-2, 0)) => assert!(!node.is_leaf()),
                Some((-2, 1)) => assert!(!node.is_leaf()),
                _ => {}
            };
        }
    }

    #[test]
    fn insertion_tree_structure_condition() {
        let data = vec![0.49, 0.491, -0.49, 0.0];

        let point_cloud = Arc::new(DefaultCloud::<L2>::new(data, 1).unwrap());
        let builder = CoverTreeBuilder {
            scale_base: 2.0,
            leaf_cutoff: 1,
            min_res_index: -9,
            use_singletons: true,
            verbosity: 0,
            partition_type: PartitionType::First,
            rng_seed: Some(0),
        };
        let tree = builder.build(point_cloud).unwrap();
        let reader = tree.reader();

        println!("Testing top layer");
        let top_layer = reader.layer(-1);
        println!("Should only be one node");
        assert!(top_layer.len() == 1);
        println!("The root should not be a leaf");
        assert!(reader
            .get_node_and((-1, 3).into(), |n| !n.is_leaf())
            .unwrap());
        println!("The root should have children");
        assert!(reader
            .get_node_and((-1, 3).into(), |n| n.children().is_some())
            .unwrap());

        println!("Testing Mid Layer");
        let mid_layer = reader.layer(-2);
        println!("Should have 2 nodes");
        assert!(mid_layer.len() == 2);
        println!("Nested child of root should leafify");
        assert!(reader
            .get_node_and((-2, 3).into(), |n| n.is_leaf())
            .unwrap());
        println!("Nested child of root should not have any children");
        assert!(reader
            .get_node_and((-2, 3).into(), |n| n.children().is_none())
            .unwrap());
        println!("-0.49 is a singleton that shouldn't be here.");
        assert!(reader
            .get_node_and((-2, 2).into(), |n| n.is_leaf())
            .is_none());
        assert!(reader.no_dangling_refs());
    }

    #[test]
    fn singleltons_off_condition() {
        let data = vec![0.49, 0.491, -0.49, 0.0];

        let point_cloud = Arc::new(DefaultCloud::<L2>::new(data, 1).unwrap());

        let builder = CoverTreeBuilder {
            scale_base: 2.0,
            leaf_cutoff: 1,
            min_res_index: -9,
            use_singletons: false,
            verbosity: 0,
            partition_type: PartitionType::First,
            rng_seed: Some(0),
        };
        let tree = builder.build(point_cloud).unwrap();
        let reader = tree.reader();

        println!("-0.49 is a singleton that should be here.");
        assert!(reader
            .get_node_and((-2, 2).into(), |n| n.is_leaf())
            .is_some());
        assert!(reader.no_dangling_refs());
    }
}
