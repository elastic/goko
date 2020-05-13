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

use crate::plugins::TreePluginSet;
use crate::*;
use data_caches::*;
use layer::*;
use node::*;
use pbr::ProgressBar;
//use pointcloud::*;
use std::cmp::{max, min};
use std::sync::{atomic, Arc, RwLock};

use crossbeam_channel::{unbounded, Receiver, Sender};
use errors::GrandmaResult;

use std::time::Instant;

#[derive(Debug)]
struct BuilderNode {
    scale_index: i32,
    covered: CoveredData,
}

type NodeSplitResult<M> = GrandmaResult<(i32, PointIndex, CoverNode<M>)>;

impl BuilderNode {
    fn new<M: Metric>(parameters: &CoverTreeParameters<M>) -> GrandmaResult<BuilderNode> {
        let covered = CoveredData::new(&parameters.point_cloud)?;
        let scale_index = (covered.max_distance()).log(parameters.scale_base).ceil() as i32;
        Ok(BuilderNode {
            scale_index,
            covered,
        })
    }

    #[inline]
    fn address(&self) -> NodeAddress {
        (self.scale_index, self.covered.center_index)
    }

    fn split_parallel<M: Metric>(
        self,
        parameters: &Arc<CoverTreeParameters<M>>,
        node_sender: &Arc<Sender<NodeSplitResult<M>>>,
    ) {
        let parameters = Arc::clone(parameters);
        let node_sender = Arc::clone(node_sender);
        rayon::spawn(move || {
            let (si, pi) = self.address();
            match self.split(&parameters) {
                Ok((new_node, mut new_nodes)) => {
                    node_sender.send(Ok((si, pi, new_node))).unwrap();
                    while let Some(node) = new_nodes.pop() {
                        node.split_parallel(&parameters, &node_sender);
                    }
                }
                Err(e) => node_sender.send(Err(e)).unwrap(),
            };
        });
    }

    fn split<M: Metric>(
        self,
        parameters: &Arc<CoverTreeParameters<M>>,
    ) -> GrandmaResult<(CoverNode<M>, Vec<BuilderNode>)> {
        //println!("=====================");
        //println!("Splitting node with address {:?} and covered: {:?}", self.address(),self.covered);

        let scale_index = self.scale_index;
        let covered = self.covered;
        let mut node = CoverNode::new((scale_index, covered.center_index));
        let radius = covered.max_distance();
        let mut new_nodes = Vec::new();
        node.set_radius(radius);
        /* Occasionally there's a small cluster split off of at a low min_res_index.
        This brings the scale-index down/min_res_index up quickly, locally.
        */
        if covered.len() <= parameters.leaf_cutoff || scale_index < parameters.min_res_index {
            //println!("== This is getting cut down by parameters ==");
            node.insert_singletons(covered.into_indexes());
        } else {
            let next_scale_index = min(
                scale_index - 1,
                max(
                    radius.log(parameters.scale_base).ceil() as i32,
                    parameters.min_res_index,
                ),
            );
            let next_scale = parameters.scale_base.powi(next_scale_index);

            /*
            We get a bunch of points (close) that are within `next_scale` of the center.
            we also get a bunch of points further out (new_fars). For these we need to find centers.
            */

            let (close, mut fars) = covered.split(next_scale).unwrap();
            //println!("== Split loop setup with scale {}, and scale index {} ==", next_scale, next_scale_index);
            //println!("\tCovered: {:?}", close);
            //println!("\tNot Covered: {:?}", fars);

            node.insert_nested_child(next_scale_index, close.len())?;
            let new_node = BuilderNode {
                scale_index: next_scale_index,
                covered: close,
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
                let new_close = fars.pick_center(next_scale, &parameters.point_cloud)?;
                //println!("\t\t [{}] New Covered: {:?}",split_count, new_close);
                if new_close.len() == 1 && parameters.use_singletons {
                    /*
                    We have a vast quantity of internal ourliers. These are singleton points that are
                    at least next_scale away from each other. These could be fully fledged leaf nodes,
                    or we can short circut them and just store a reference.

                    On malware data 80% of the data are outliers of this type. References are a significant
                    ram savings.
                    */
                    node.insert_singleton(new_close.center_index);
                } else {
                    node.insert_child((next_scale_index, new_close.center_index), new_close.len())?;
                    let new_node = BuilderNode {
                        scale_index: next_scale_index,
                        covered: new_close,
                    };
                    new_nodes.push(new_node);
                    parameters
                        .total_nodes
                        .fetch_add(1, atomic::Ordering::SeqCst);
                }
            }
        }

        if new_nodes.len() == 1 && new_nodes[0].covered.len() == 1 {
            node.remove_children();
            parameters
                .total_nodes
                .fetch_sub(1, atomic::Ordering::SeqCst);
            node.insert_singletons(new_nodes.pop().unwrap().covered.into_indexes());
        }

        node.update_metasummary(&parameters.point_cloud)?;
        // This node is done, send it in
        //println!("=====================");
        Ok((node, new_nodes))
    }
}

/// A construction object for a covertree.
#[derive(Debug, Default)]
pub struct CoverTreeBuilder {
    /// See paper or main description, governs the number of children of each node. Higher is more.
    pub scale_base: f32,
    /// If a node covers less than or equal to this number of points, it becomes a leaf.
    pub leaf_cutoff: usize,
    /// If a node has scale index less than or equal to this, it becomes a leaf
    pub min_res_index: i32,
    /// If you don't want singletons messing with your tree and want everything to be a node or a element of leaf node, make this true.
    pub use_singletons: bool,
    /// Printing verbosity. 2 is the default and gives a progress bar. Still not fully pulled thru the codebase.
    /// This should be replaced by a logging solution
    pub verbosity: u32,
}

impl CoverTreeBuilder {
    /// Creates a new builder with sensible defaults.
    pub fn new() -> CoverTreeBuilder {
        CoverTreeBuilder {
            scale_base: 2.0,
            leaf_cutoff: 1,
            min_res_index: -10,
            use_singletons: true,
            verbosity: 2,
        }
    }

    ///
    pub fn set_scale_base(&mut self, x: f32) -> &mut Self {
        self.scale_base = x;
        self
    }
    ///
    pub fn set_leaf_cutoff(&mut self, x: usize) -> &mut Self {
        self.leaf_cutoff = x;
        self
    }
    ///
    pub fn set_min_res_index(&mut self, x: i32) -> &mut Self {
        self.min_res_index = x;
        self
    }
    ///
    pub fn set_use_singletons(&mut self, x: bool) -> &mut Self {
        self.use_singletons = x;
        self
    }
    ///
    pub fn set_verbosity(&mut self, x: u32) -> &mut Self {
        self.verbosity = x;
        self
    }
    /// Pass a point cloud object when ready.
    /// To do, make this point cloud an Arc
    pub fn build<M: Metric>(
        &self,
        point_cloud: PointCloud<M>,
    ) -> GrandmaResult<CoverTreeWriter<M>> {
        let parameters = CoverTreeParameters {
            total_nodes: atomic::AtomicUsize::new(1),
            scale_base: self.scale_base,
            leaf_cutoff: self.leaf_cutoff,
            min_res_index: self.min_res_index,
            use_singletons: self.use_singletons,
            point_cloud,
            verbosity: self.verbosity,
            plugins: RwLock::new(TreePluginSet::new()),
        };

        let root = BuilderNode::new(&parameters)?;
        let root_address = root.address();
        let scale_range = root_address.0 - parameters.min_res_index;
        let mut layers = Vec::with_capacity(scale_range as usize);
        layers.push(CoverLayerWriter::new(parameters.min_res_index - 1));
        for i in 0..(scale_range + 1) {
            layers.push(CoverLayerWriter::new(parameters.min_res_index + i as i32));
        }

        let (node_sender, node_receiver): (
            Sender<NodeSplitResult<M>>,
            Receiver<NodeSplitResult<M>>,
        ) = unbounded();

        let node_sender = Arc::new(node_sender);
        let parameters = Arc::new(parameters);
        root.split_parallel(&parameters, &node_sender);
        let mut pb = ProgressBar::new(1u64);
        if parameters.verbosity > 1 {
            pb.format("╢▌▌░╟");
        }

        let mut cover_tree = CoverTreeWriter {
            parameters: Arc::clone(&parameters),
            layers,
            root_address,
        };

        let mut inserted_nodes: usize = 0;
        let now = Instant::now();
        loop {
            if let Ok(res) = node_receiver.recv() {
                let (scale_index, point_index, new_node) = res.unwrap();
                unsafe {
                    cover_tree.insert_raw(scale_index, point_index, new_node);
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
        data: Box<[f32]>,
        data_dim: usize,
        labels: Box<[f32]>,
        labels_dim: usize,
    ) -> Arc<CoverTreeParameters<L2>> {
        let point_cloud =
            PointCloud::<L2>::simple_from_ram(data, data_dim, labels, labels_dim).unwrap();
        Arc::new(CoverTreeParameters {
            total_nodes: atomic::AtomicUsize::new(1),
            scale_base: 2.0,
            leaf_cutoff: 0,
            min_res_index: -9,
            use_singletons: true,
            point_cloud,
            verbosity: 0,
            plugins: RwLock::new(TreePluginSet::new()),
        })
    }

    #[test]
    fn splits_conditions() {
        let mut data = Vec::with_capacity(20);
        for _i in 0..19 {
            data.push(rand::random::<f32>());
        }
        data.push(0.0);

        let test_parameters =
            create_test_parameters(Box::from(data.clone()), 1, Box::from(data.clone()), 1);
        let build_node = BuilderNode::new(&test_parameters).unwrap();
        let (scale_index, center_index) = build_node.address();

        println!("{:?}", build_node);
        println!(
            "The center_index for the covered data should be 19 but is {}",
            build_node.covered.center_index
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
    fn tree_structure_condition() {
        let data = vec![0.49, 0.491, -0.49, 0.0];
        let labels = vec![0.0, 0.0, 1.0, 1.0];

        let test_parameters =
            create_test_parameters(Box::from(data.clone()), 1, Box::from(labels), 1);
        let build_node = BuilderNode::new(&test_parameters).unwrap();

        let (node_sender, node_receiver): (
            Sender<GrandmaResult<(i32, PointIndex, CoverNode<L2>)>>,
            Receiver<GrandmaResult<(i32, PointIndex, CoverNode<L2>)>>,
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
            let (scale_index, center_index, node) = pat.unwrap();
            println!("{:?}", node);
            match (scale_index, center_index) {
                (-1, 3) => assert!(!node.is_leaf()),
                (-2, 3) => assert!(node.is_leaf()),
                (-2, 2) => assert!(node.is_leaf()),
                (-2, 0) => assert!(!node.is_leaf()),
                (-2, 1) => assert!(!node.is_leaf()),
                _ => {}
            };
        }
    }

    #[test]
    fn insertion_tree_structure_condition() {
        let data = vec![0.49, 0.491, -0.49, 0.0];
        let labels = vec![0.0, 0.0, 1.0, 1.0];

        let point_cloud =
            PointCloud::<L2>::simple_from_ram(Box::from(data), 1, Box::from(labels), 1).unwrap();
        let builder = CoverTreeBuilder {
            scale_base: 2.0,
            leaf_cutoff: 1,
            min_res_index: -9,
            use_singletons: true,
            verbosity: 0,
        };
        let tree = builder.build(point_cloud).unwrap();
        let reader = tree.reader();

        println!("Testing top layer");
        let top_layer = reader.layer(-1);
        println!("Should only be one node");
        assert!(top_layer.node_count() == 1);
        println!("The root should not be a leaf");
        assert!(reader.get_node_and((-1, 3), |n| !n.is_leaf()).unwrap());
        println!("The root should have children");
        assert!(reader
            .get_node_and((-1, 3), |n| n.children().is_some())
            .unwrap());

        println!("Testing Mid Layer");
        let mid_layer = reader.layer(-2);
        println!("Should have 2 nodes");
        assert!(mid_layer.node_count() == 2);
        println!("Nested child of root should leafify");
        assert!(reader.get_node_and((-2, 3), |n| n.is_leaf()).unwrap());
        println!("Nested child of root should not have any children");
        assert!(reader
            .get_node_and((-2, 3), |n| n.children().is_none())
            .unwrap());
        println!("-0.49 is a singleton that shouldn't be here.");
        assert!(reader.get_node_and((-2, 2), |n| n.is_leaf()).is_none());
        assert!(reader.no_dangling_refs());
    }

    #[test]
    fn singleltons_off_condition() {
        let data = vec![0.49, 0.491, -0.49, 0.0];
        let labels = vec![0.0, 0.0, 1.0, 1.0];

        let point_cloud =
            PointCloud::<L2>::simple_from_ram(Box::from(data), 1, Box::from(labels), 1).unwrap();
        let builder = CoverTreeBuilder {
            scale_base: 2.0,
            leaf_cutoff: 1,
            min_res_index: -9,
            use_singletons: false,
            verbosity: 0,
        };
        let tree = builder.build(point_cloud).unwrap();
        let reader = tree.reader();

        println!("-0.49 is a singleton that should be here.");
        assert!(reader.get_node_and((-2, 2), |n| n.is_leaf()).is_some());
        assert!(reader.no_dangling_refs());
    }
}
