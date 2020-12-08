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

//! # The Cover Tree Data Structure
//! To keep a no-lock, yet editable cover tree that can be queried in parallel we need to keep a pair of hash-maps for each layer.
//! They are duplicated (this is slow, it should be changed to an unsafe partial duplication). All the readers are pointed at one
//! hash-map on each layer, while this write head is pointed at the other. All writes to the write head are not available to the
//! readers until refresh is called. Each write is queued for a pair of write operations, the first is to the hash-maps available
//! to the write head, then after refresh this queue is drained and the second write operation is performed on the other hash-maps.
//!
//! To ensure consistency only call refresh when you have a valid tree. For example if you are removing a subtree starting from some root node
//! only call refresh once you're finished.
//!
//! The covertree is meant to be eventually consistent with no mutexes or any other locks. To accomplish this there
//! is a reader head and a writer head. The reader head is read only and has access to the most recent "valid" tree.
//! For now, valid only means a *weak covertree*.
//!
//! The hashmap pair idea is in `layer` and originally comes from Jon Gjengset.

use super::layer::*;
use super::node::*;
use crate::*;
//use pointcloud::*;

use crate::monomap::{MonoReadHandle, MonoWriteHandle};
use crate::tree_file_format::*;
use std::sync::{atomic, Arc, RwLock};

use super::query_tools::{KnnQueryHeap, RoutingQueryHeap};
use crate::plugins::{GokoPlugin, TreePluginSet};
use errors::{GokoError, GokoResult};
use std::iter::Iterator;
use std::iter::Rev;
use std::ops::Range;
use std::slice::Iter;

use plugins::labels::*;

/// When 2 spheres overlap under a node, and there is a point in the overlap we have to decide
/// to which sphere it belongs. As we create the nodes in a particular sequence, we can assign them
/// to the first to be created or we can assign it to the nearest.
#[derive(Debug, Copy, Clone)]
pub enum PartitionType {
    /// Conflicts assigning a point to several eligible nodes are assigned to the nearest node.
    Nearest,
    /// Conflicts assigning a point to several eligible nodes are assigned to the first node to be created.
    First,
}

/// Container for the parameters governing the construction of the covertree
#[derive(Debug)]
pub struct CoverTreeParameters<D: PointCloud> {
    /// An atomic that tracks all nodes as they are created across all threads.
    /// This may not reflect what your current reader can see.
    pub total_nodes: atomic::AtomicUsize,
    /// See paper or main description, governs the number of children of each node. Higher is more.
    pub scale_base: f32,
    /// If a node covers less than or equal to this number of points, it becomes a leaf.
    pub leaf_cutoff: usize,
    /// If a node has scale index less than or equal to this, it becomes a leaf
    pub min_res_index: i32,
    /// If you don't want singletons messing with your tree and want everything to be a node or a element of leaf node, make this true.
    pub use_singletons: bool,
    /// The partition type of the tree
    pub partition_type: PartitionType,
    /// The point cloud this tree references
    pub point_cloud: Arc<D>,
    /// This should be replaced by a logging solution
    pub verbosity: u32,
    /// The seed to use for deterministic trees. This is xor-ed with the point index to create a seed for `rand::rngs::SmallRng`.
    /// 
    /// Pass in None if you want to use the host os's entropy instead. 
    pub rng_seed: Option<u64>,
    /// This is where the base plugins are are stored.
    pub plugins: RwLock<TreePluginSet>,
}

impl<D: PointCloud> CoverTreeParameters<D> {
    /// Gets the index of the layer in the vector.
    #[inline]
    pub fn internal_index(&self, scale_index: i32) -> usize {
        if scale_index < self.min_res_index {
            0
        } else {
            (scale_index - self.min_res_index + 1) as usize
        }
    }
}

/// Helper struct for iterating thru the reader's of the the layers.
pub type LayerIter<'a, D> = Rev<std::iter::Zip<Range<i32>, Iter<'a, CoverLayerReader<D>>>>;

/// # Cover Tree Reader Head
///
/// You can clone the reader head, though this is a relatively expensive operation and should not be performed lightly.
///
/// All queries of the covertree should go through a reader head. This includes queries you are doing to modify the tree.
/// There are no thread locks anywhere in the code below the reader head, so it's fast.
///
/// The data structure is just a list of `CoverLayerReader`s, the parameter's object and the root address. Copies are relatively
/// expensive as each `CoverLayerReader` contains several Arcs that need to be cloned.
pub struct CoverTreeReader<D: PointCloud> {
    parameters: Arc<CoverTreeParameters<D>>,
    layers: Vec<CoverLayerReader<D>>,
    root_address: NodeAddress,
    final_addresses: MonoReadHandle<usize, NodeAddress>,
}

impl<D: PointCloud> Clone for CoverTreeReader<D> {
    fn clone(&self) -> CoverTreeReader<D> {
        CoverTreeReader {
            parameters: self.parameters.clone(),
            layers: self.layers.clone(),
            root_address: self.root_address,
            final_addresses: self.final_addresses.clone(),
        }
    }
}

impl<D: PointCloud + LabeledCloud> CoverTreeReader<D> {
    /// Reads the contents of a plugin, due to the nature of the plugin map we have to access it with a
    /// closure.
    pub fn get_node_label_summary(
        &self,
        node_address: (i32, usize),
    ) -> Option<Arc<SummaryCounter<D::LabelSummary>>> {
        self.layers[self.parameters.internal_index(node_address.0)]
            .get_node_and(node_address.1, |n| n.label_summary())
            .flatten()
    }
}

impl<D: PointCloud + MetaCloud> CoverTreeReader<D> {
    /// Reads the contents of a plugin, due to the nature of the plugin map we have to access it with a
    /// closure.
    pub fn get_node_metasummary(
        &self,
        node_address: (i32, usize),
    ) -> Option<Arc<SummaryCounter<D::MetaSummary>>> {
        self.layers[self.parameters.internal_index(node_address.0)]
            .get_node_and(node_address.1, |n| n.metasummary())
            .flatten()
    }
}

impl<D: PointCloud> CoverTreeReader<D> {
    /// A reference to the point cloud the tree was built on.
    pub fn point_cloud(&self) -> &Arc<D> {
        &self.parameters.point_cloud
    }

    /// Returns a borrowed reader for a cover layer.
    ///
    pub fn layer(&self, scale_index: i32) -> &CoverLayerReader<D> {
        &self.layers[self.parameters.internal_index(scale_index)]
    }

    /// simple helper to get the scale from the scale index and the scale base, this is just `b^i`
    pub fn scale(&self, scale_index: i32) -> f32 {
        self.parameters.scale_base.powi(scale_index)
    }

    /// Read only access to the internals of a node.
    pub fn get_node_and<F, T>(&self, node_address: (i32, usize), f: F) -> Option<T>
    where
        F: FnOnce(&CoverNode<D>) -> T,
    {
        self.layers[self.parameters.internal_index(node_address.0)]
            .get_node_and(node_address.1, |n| f(n))
    }

    /// Grabs all children indexes and allows you to query against them. Usually used at the tree level so that you
    /// can access the child nodes as they are not on this layer.
    pub fn get_node_children_and<F, T>(&self, node_address: (i32, usize), f: F) -> Option<T>
    where
        F: FnOnce(NodeAddress, &[NodeAddress]) -> T,
    {
        self.layers[self.parameters.internal_index(node_address.0)]
            .get_node_children_and(node_address.1, f)
    }

    /// The root of the tree. Pass this to `get_node_and` to get the root node's content and start a traversal of the tree.
    pub fn root_address(&self) -> NodeAddress {
        self.root_address
    }

    /// An iterator for accessing the layers starting from the layer who holds the root.
    pub fn layers(&self) -> LayerIter<D> {
        ((self.parameters.min_res_index - 1)
            ..(self.layers.len() as i32 + self.parameters.min_res_index - 1))
            .zip(self.layers.iter())
            .rev()
    }

    /// Returns the number of layers in the tree. This is _not_ the number of non-zero layers.
    pub fn len(&self) -> usize {
        self.layers.len()
    }

    /// Returns the number of layers in the tree. This is _not_ the number of non-zero layers.
    pub fn is_empty(&self) -> bool {
        self.layers.is_empty()
    }

    /// If you want to build a new tree with shared parameters, this is helpful.
    pub fn parameters(&self) -> &Arc<CoverTreeParameters<D>> {
        &self.parameters
    }

    /// This is the total number of nodes in the tree. This queries each layer, so it's not a simple return int.
    pub fn node_count(&self) -> usize {
        self.layers().fold(0, |a, (_si, l)| a + l.len())
    }

    /// Returns the scale index range. It starts at the minimum min_res_index and ends at the top. You can reverse this for the correct order.
    pub fn scale_range(&self) -> Range<i32> {
        (self.parameters.min_res_index)
            ..(self.parameters.min_res_index - 1 + self.layers.len() as i32)
    }

    /// Access the stored tree plugin
    pub fn get_plugin_and<T: Send + Sync + 'static, F, S>(&self, transform_fn: F) -> Option<S>
    where
        F: FnOnce(&T) -> S,
    {
        self.parameters
            .plugins
            .read()
            .unwrap()
            .get::<T>()
            .map(transform_fn)
    }

    /// Reads the contents of a plugin, due to the nature of the plugin map we have to access it with a
    /// closure.
    pub fn get_node_plugin_and<T: Send + Sync + 'static, F, S>(
        &self,
        node_address: (i32, usize),
        transform_fn: F,
    ) -> Option<S>
    where
        F: FnOnce(&T) -> S,
    {
        self.layers[self.parameters.internal_index(node_address.0)]
            .get_node_and(node_address.1, |n| n.get_plugin_and(transform_fn))
            .flatten()
    }

    /// # The KNN query.
    /// This works by recursively greedily querying the nearest child node with the lowest scale index to the point in question of a node,
    /// starting at the root until we hit a leaf. During this process all nodes touched are pushed onto a pair of min-heaps, one
    /// to keep track of the nodes' who have been not yet been queried for their children or singletons (called the `child_heap`, and
    /// the other to track the nodes who have not yet been queried for their singletons (called the `singleton_heap`). Both these heaps are
    /// min-heaps, ordering the nodes lexicographically by minimum possible distance to the point, then scale index, and finally the
    /// actual distance to the query point.
    ///
    /// Once we reach the bottom we pop a node from the `singleton_heap` and if that node could have a point within range we query that
    /// node's singletons. These should be the closest to the query point.
    /// We then pop a node from the `child_heap` and repeat the greedy query starting from the popped node and terminating at a leaf.
    ///
    /// The process terminates when there is no node that could cover a point in the tree closer than the furthest point we already have in
    /// our KNN.
    ///
    /// See `query_tools::KnnQueryHeap` for the pair of heaps and mechanisms for tracking the minimum distance and the current knn set.
    /// See the `nodes::CoverNode::singleton_knn` and `nodes::CoverNode::child_knn` for the brute force node based knn.
    pub fn knn<'a>(&self, point: &D::PointRef<'a>, k: usize) -> GokoResult<Vec<(f32, usize)>> {
        let mut query_heap = KnnQueryHeap::new(k, self.parameters.scale_base);

        let root_center = self.parameters.point_cloud.point(self.root_address.1)?;
        let dist_to_root = D::Metric::dist(&root_center, &point);
        query_heap.push_nodes(&[self.root_address], &[dist_to_root], None);
        self.greedy_knn_nodes(&point, &mut query_heap);

        while let Some((_dist, address)) = query_heap.closest_unvisited_singleton_covering_address()
        {
            self.get_node_and(address, |n| {
                n.singleton_knn(&point, &self.parameters.point_cloud, &mut query_heap)
            });
            self.greedy_knn_nodes(&point, &mut query_heap);
        }

        Ok(query_heap.unpack())
    }

    /// Same as knn, but only deals with non-singleton points
    pub fn routing_knn<'a>(
        &self,
        point: &D::PointRef<'a>,
        k: usize,
    ) -> GokoResult<Vec<(f32, usize)>> {
        let mut query_heap = KnnQueryHeap::new(k, self.parameters.scale_base);

        let root_center = self.parameters.point_cloud.point(self.root_address.1)?;
        let dist_to_root = D::Metric::dist(&root_center, &point);
        query_heap.push_nodes(&[self.root_address], &[dist_to_root], None);
        self.greedy_knn_nodes(point, &mut query_heap);

        while self.greedy_knn_nodes(point, &mut query_heap) {}
        Ok(query_heap.unpack())
    }

    fn greedy_knn_nodes<'a>(&self, point: &D::PointRef<'a>, query_heap: &mut KnnQueryHeap) -> bool {
        let mut did_something = false;
        while let Some((dist, nearest_address)) =
            query_heap.closest_unvisited_child_covering_address()
        {
            if self
                .get_node_and(nearest_address, |n| n.is_leaf())
                .unwrap_or(true)
            {
                break;
            } else {
                self.get_node_and(nearest_address, |n| {
                    n.child_knn(Some(dist), &point, &self.parameters.point_cloud, query_heap)
                });
            }
            did_something = true;
        }
        did_something
    }

    /// # Dry Insert Query
    pub fn path<'a>(&self, point: &D::PointRef<'a>) -> GokoResult<Vec<(f32, NodeAddress)>> {
        let root_center = self.parameters.point_cloud.point(self.root_address.1)?;
        let mut current_distance = D::Metric::dist(&root_center, &point);
        let mut current_address = self.root_address;
        let mut trace = vec![(current_distance, current_address)];
        while let Some(nearest) =
            self.get_node_and(current_address, |n| match self.parameters.partition_type {
                PartitionType::Nearest => n.nearest_covering_child(
                    self.parameters.scale_base,
                    current_distance,
                    point,
                    &self.parameters.point_cloud,
                ),
                PartitionType::First => n.first_covering_child(
                    self.parameters.scale_base,
                    current_distance,
                    point,
                    &self.parameters.point_cloud,
                ),
            })
        {
            if let Some(nearest) = nearest? {
                trace.push(nearest);
                current_distance = nearest.0;
                current_address = nearest.1;
            } else {
                break;
            }
        }
        Ok(trace)
    }

    ///
    pub fn known_path(&self, point_index: usize) -> GokoResult<Vec<(f32, NodeAddress)>> {
        self.final_addresses
            .get_and(&point_index, |addr| {
                let mut path = Vec::with_capacity((self.root_address().0 - addr.0) as usize);
                let mut parent = Some(*addr);
                while let Some(addr) = parent {
                    path.push(addr);
                    parent = self.get_node_and(addr, |n| n.parent_address()).flatten();
                }
                (&mut path[..]).reverse();
                let point_indexes: Vec<usize> = path.iter().map(|na| na.1).collect();
                let dists = self
                    .parameters
                    .point_cloud
                    .distances_to_point_index(point_index, &point_indexes[..])
                    .unwrap();
                dists.iter().zip(path).map(|(d, a)| (*d, a)).collect()
            })
            .ok_or(GokoError::IndexNotInTree(point_index))
    }

    ///Computes the fractal dimension of a node
    pub fn node_fractal_dim(&self, node_address: NodeAddress) -> f32 {
        let count: f32 = self
            .get_node_and(node_address, |n| {
                (n.singletons_len() + n.children_len()) as f32
            })
            .unwrap() as f32;
        count.log(self.parameters.scale_base)
    }

    ///Computes the weighted fractal dimension of a node
    pub fn node_weighted_fractal_dim(&self, node_address: NodeAddress) -> f32 {
        let weighted_count: f32 = self
            .get_node_and(node_address, |n| {
                let singleton_count = n.singletons().len() as f32;
                let mut max_pop: usize = 1;
                let mut weighted_count: f32 = 0.0;
                if let Some((nested_scale, children)) = n.children() {
                    let mut pops: Vec<usize> = children
                        .iter()
                        .map(|child_addr| {
                            self.get_node_and(*child_addr, |child| child.coverage_count())
                                .unwrap()
                        })
                        .collect();
                    pops.push(
                        self.get_node_and((nested_scale, node_address.1), |child| {
                            child.coverage_count()
                        })
                        .unwrap(),
                    );
                    max_pop = *pops.iter().max().unwrap();
                    pops.iter()
                        .for_each(|p| weighted_count += (*p as f32) / (max_pop as f32));
                }
                weighted_count + singleton_count / (max_pop as f32)
            })
            .unwrap();
        weighted_count.log(self.parameters.scale_base)
    }

    ///Computes the fractal dimension of a layer
    pub fn layer_fractal_dim(&self, scale_index: i32) -> f32 {
        let parent_layer = self.layer(scale_index);
        let parent_count = parent_layer.len() as f32;
        let mut child_count: f32 = 0.0;
        parent_layer
            .for_each_node(|_, n| child_count += (n.singletons_len() + n.children_len()) as f32);
        child_count.log(self.parameters.scale_base) - parent_count.log(self.parameters.scale_base)
    }

    ///Computes the weighted fractal dimension of a node
    pub fn layer_weighted_fractal_dim(&self, scale_index: i32) -> f32 {
        // gather the coverages of every node on the layer and their's children
        let parent_layer = self.layer(scale_index);
        let mut parent_coverage_counts: Vec<usize> = Vec::new();
        let mut child_coverage_counts: Vec<usize> = Vec::new();
        let mut singletons_count: f32 = 0.0;
        parent_layer.for_each_node(|center_index, n| {
            parent_coverage_counts.push(n.coverage_count());

            singletons_count += n.singletons().len() as f32;
            if let Some((nested_scale, children)) = n.children() {
                child_coverage_counts.extend(children.iter().map(|child_addr| {
                    self.get_node_and(*child_addr, |child| child.coverage_count())
                        .unwrap()
                }));
                child_coverage_counts.push(
                    self.get_node_and((nested_scale, *center_index), |child| {
                        child.coverage_count()
                    })
                    .unwrap(),
                );
            }
        });
        // Get the maximum count
        let max_parent_pop: f32 = *parent_coverage_counts.iter().max().unwrap_or(&1) as f32;
        let max_child_pop: f32 = *child_coverage_counts.iter().max().unwrap_or(&1) as f32;

        // Normalize the counts by the maximum
        let weighted_child_sum: f32 = singletons_count / max_child_pop
            + child_coverage_counts
                .iter()
                .fold(0.0, |a, c| a + (*c as f32) / max_child_pop);
        let weighted_parent_sum: f32 = parent_coverage_counts
            .iter()
            .fold(0.0, |a, c| a + (*c as f32) / max_parent_pop);

        // take the log and return
        weighted_child_sum.log(self.parameters.scale_base)
            - weighted_parent_sum.log(self.parameters.scale_base)
    }

    /// Checks that there are no node addresses in the child list of any node that don't reference a node in the tree.
    /// Please calmly panic if there are, the tree is very invalid.
    pub(crate) fn no_dangling_refs(&self) -> bool {
        let mut refs_to_check = vec![self.root_address];
        while let Some(node_addr) = refs_to_check.pop() {
            println!("checking {:?}", node_addr);
            println!("refs_to_check: {:?}", refs_to_check);
            let node_exists = self.get_node_and(node_addr, |n| {
                if let Some((nested_scale, other_children)) = n.children() {
                    println!(
                        "Pushing: {:?}, {:?}",
                        (nested_scale, other_children),
                        other_children
                    );
                    refs_to_check.push((nested_scale, node_addr.1));
                    refs_to_check.extend(&other_children[..]);
                }
            });
            if node_exists.is_none() {
                return false;
            }
        }
        true
    }
}

///
pub struct CoverTreeWriter<D: PointCloud> {
    pub(crate) parameters: Arc<CoverTreeParameters<D>>,
    pub(crate) layers: Vec<CoverLayerWriter<D>>,
    pub(crate) root_address: NodeAddress,
    pub(crate) final_addresses: MonoWriteHandle<usize, NodeAddress>,
}

impl<D: PointCloud + LabeledCloud> CoverTreeWriter<D> {
    ///
    pub fn generate_summaries(&mut self) {
        self.add_plugin::<LabelSummaryPlugin>(LabelSummaryPlugin::default())
    }
}

impl<D: PointCloud + MetaCloud> CoverTreeWriter<D> {
    ///
    pub fn generate_meta_summaries(&mut self) {
        self.add_plugin::<MetaSummaryPlugin>(MetaSummaryPlugin::default())
    }
}

impl<D: PointCloud> CoverTreeWriter<D> {
    ///
    pub fn add_plugin<P: GokoPlugin<D>>(&mut self, plug_in: P) {
        P::prepare_tree(&plug_in, self);
        let reader = self.reader();
        for layer in self.layers.iter_mut() {
            layer.reader().for_each_node(|pi, n| {
                if let Some(node_component) = P::node_component(&plug_in, n, &reader) {
                    unsafe {
                        layer.update_node(*pi, move |n| n.insert_plugin(node_component.clone()))
                    }
                }
            });
            layer.refresh()
        }
        self.parameters.plugins.write().unwrap().insert(plug_in);
    }

    /// Provides a reference to a `CoverLayerWriter`. Do not use, unless you're going to leave the tree in a *valid* state.
    pub(crate) unsafe fn layer(&mut self, scale_index: i32) -> &mut CoverLayerWriter<D> {
        &mut self.layers[self.parameters.internal_index(scale_index)]
    }

    pub(crate) unsafe fn update_node<F>(&mut self, address: NodeAddress, update_fn: F)
    where
        F: Fn(&mut CoverNode<D>) + 'static + Send + Sync,
    {
        self.layers[self.parameters.internal_index(address.0)].update_node(address.1, update_fn);
    }

    /// Creates a reader for queries.
    pub fn reader(&self) -> CoverTreeReader<D> {
        CoverTreeReader {
            parameters: Arc::clone(&self.parameters),
            layers: self.layers.iter().map(|l| l.reader()).collect(),
            root_address: self.root_address,
            final_addresses: self.final_addresses.factory().handle(),
        }
    }

    pub(crate) unsafe fn insert_raw(
        &mut self,
        scale_index: i32,
        point_index: usize,
        node: CoverNode<D>,
    ) {
        self.layers[self.parameters.internal_index(scale_index)].insert_raw(point_index, node);
    }

    /// Loads a tree from a protobuf. There's a `load_tree` in `utils` that handles loading from a path to a protobuf file.
    pub fn load(cover_proto: &CoreProto, point_cloud: Arc<D>) -> GokoResult<CoverTreeWriter<D>> {
        let partition_type = if cover_proto.partition_type == "first" {
            PartitionType::First
        } else {
            PartitionType::Nearest
        };

        let parameters = Arc::new(CoverTreeParameters {
            total_nodes: atomic::AtomicUsize::new(0),
            use_singletons: cover_proto.use_singletons,
            scale_base: cover_proto.scale_base as f32,
            leaf_cutoff: cover_proto.cutoff as usize,
            min_res_index: cover_proto.resolution as i32,
            point_cloud,
            verbosity: 2,
            partition_type,
            plugins: RwLock::new(TreePluginSet::new()),
            rng_seed: None,
        });
        let root_address = (
            cover_proto.get_root_scale(),
            cover_proto.get_root_index() as usize,
        );
        let layers: Vec<CoverLayerWriter<D>> = cover_proto
            .get_layers()
            .par_iter()
            .map(|l| CoverLayerWriter::load(l))
            .collect();

        let (_final_addresses_reader, final_addresses) = monomap::new();

        let mut tree = CoverTreeWriter {
            parameters,
            layers,
            root_address,
            final_addresses,
        };

        tree.refresh_final_indexes();

        Ok(tree)
    }

    /// Completely redoes the final index map.
    pub fn refresh_final_indexes(&mut self) {
        let reader = self.reader();
        let mut unvisited_nodes: Vec<NodeAddress> = vec![self.root_address];
        while !unvisited_nodes.is_empty() {
            let cur_add = unvisited_nodes.pop().unwrap();
            reader
                .get_node_and(cur_add, |n| {
                    for singleton in n.singletons() {
                        self.final_addresses.insert(*singleton, cur_add);
                    }
                    if let Some((nested_si, child_addresses)) = n.children() {
                        unvisited_nodes.extend(child_addresses);
                        unvisited_nodes.push((nested_si, cur_add.1));
                    } else {
                        self.final_addresses.insert(cur_add.1, cur_add);
                    }
                })
                .unwrap();
        }

        self.final_addresses.refresh();
        self.final_addresses.refresh();
    }

    /// Encodes the tree into a protobuf. See `utils::save_tree` for saving to a file on disk.
    pub fn save(&self) -> CoreProto {
        let mut cover_proto = CoreProto::new();
        match self.parameters.partition_type {
            PartitionType::First => cover_proto.set_partition_type("first".to_string()),
            PartitionType::Nearest => cover_proto.set_partition_type("nearest".to_string()),
        }
        cover_proto.set_scale_base(self.parameters.scale_base);
        cover_proto.set_cutoff(self.parameters.leaf_cutoff as u64);
        cover_proto.set_resolution(self.parameters.min_res_index);
        cover_proto.set_use_singletons(self.parameters.use_singletons);
        cover_proto.set_dim(self.parameters.point_cloud.dim() as u64);
        cover_proto.set_count(self.parameters.point_cloud.len() as u64);
        cover_proto.set_root_scale(self.root_address.0);
        cover_proto.set_root_index(self.root_address.1 as u64);
        cover_proto.set_layers(self.layers.iter().map(|l| l.save()).collect());
        cover_proto
    }

    /// Swaps the maps on each layer so that any `CoverTreeReaders` see the updated tree.
    /// Only call once you have a valid tree.
    pub fn refresh(&mut self) {
        self.layers.iter_mut().rev().for_each(|l| l.refresh());
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    use crate::utils::cover_tree_from_labeled_yaml;
    use std::path::Path;

    pub(crate) fn build_mnist_tree() -> CoverTreeWriter<DefaultLabeledCloud<L2>> {
        let file_name = "../data/mnist_complex.yml";
        let path = Path::new(file_name);
        if !path.exists() {
            panic!(file_name.to_owned() + &" does not exist".to_string());
        }

        cover_tree_from_labeled_yaml(&path).unwrap()
    }

    pub(crate) fn build_basic_tree() -> CoverTreeWriter<DefaultLabeledCloud<L2>> {
        let data = vec![0.499, 0.49, 0.48, -0.49, 0.0];
        let labels = vec![0, 0, 0, 1, 1];

        let point_cloud = DefaultLabeledCloud::<L2>::new_simple(data, 1, labels);
        let builder = CoverTreeBuilder {
            scale_base: 2.0,
            leaf_cutoff: 1,
            min_res_index: -9,
            use_singletons: true,
            partition_type: PartitionType::Nearest,
            verbosity: 0,
            rng_seed: Some(0),
        };
        builder.build(Arc::new(point_cloud)).unwrap()
    }

    #[test]
    fn len_is_num_layers() {
        let tree = build_basic_tree();
        let reader = tree.reader();

        let mut l = 0;
        for _ in reader.layers() {
            l += 1;
        }
        assert_eq!(reader.len(), l);
    }

    #[test]
    fn layer_has_correct_scale_index() {
        let tree = build_basic_tree();
        let reader = tree.reader();
        let mut got_one = false;
        for (si, l) in reader.layers() {
            println!(
                "Scale Index, correct: {:?}, Scale Index, layer: {:?}",
                si,
                l.scale_index()
            );
            assert_eq!(si, l.scale_index());
            got_one = true;
        }
        assert!(got_one);
    }

    #[test]
    fn greedy_knn_nodes() {
        let data = vec![0.499, 0.49, 0.48, -0.49, 0.0];
        let labels = vec![0, 0, 0, 1, 1];

        let point_cloud = DefaultLabeledCloud::<L2>::new_simple(data, 1, labels);
        let builder = CoverTreeBuilder {
            scale_base: 2.0,
            leaf_cutoff: 1,
            min_res_index: -9,
            use_singletons: false,
            partition_type: PartitionType::Nearest,
            verbosity: 0,
            rng_seed: Some(0),
        };
        let tree = builder.build(Arc::new(point_cloud)).unwrap();
        let reader = tree.reader();

        let point = [-0.5];

        let mut query_heap = KnnQueryHeap::new(5, reader.parameters.scale_base);
        let dist_to_root = reader
            .parameters
            .point_cloud
            .distances_to_point(&point.as_ref(), &[reader.root_address().1])
            .unwrap()[0];
        query_heap.push_nodes(&[reader.root_address()], &[dist_to_root], None);

        assert_eq!(
            reader.root_address(),
            query_heap
                .closest_unvisited_child_covering_address()
                .unwrap()
                .1
        );

        reader.greedy_knn_nodes(&point.as_ref(), &mut query_heap);
        println!("{:#?}", query_heap);
        println!(
            "{:#?}",
            query_heap.closest_unvisited_child_covering_address()
        );
    }

    #[test]
    fn path_sanity() {
        let writer = build_basic_tree();
        let reader = writer.reader();
        let trace = reader.path(&[0.495f32].as_ref()).unwrap();
        assert!(trace.len() == 4 || trace.len() == 3);
        println!("{:?}", trace);
        for i in 0..(trace.len() - 1) {
            assert!((trace[i].1).0 > (trace[i + 1].1).0);
        }
    }

    #[test]
    fn known_path_sanity() {
        let writer = build_basic_tree();
        let reader = writer.reader();
        for i in 0..5 {
            let trace = reader.known_path(i).unwrap();
            println!("i {}, trace {:?}", i, trace);
            println!(
                "final address: {:?}",
                reader.final_addresses.get_and(&i, |i| *i)
            );
            let ad = trace.last().unwrap().1;
            reader
                .get_node_and(ad, |n| {
                    if !n.is_leaf() {
                        assert!(n.singletons().contains(&i));
                    } else {
                        assert!(
                            (ad.1 != i && n.singletons().contains(&i))
                                || (ad.1 == i && !n.singletons().contains(&i))
                        );
                    }
                })
                .unwrap();
        }
        let known_trace = reader.known_path(4).unwrap();
        let trace = reader.path(&[0.0f32].as_ref()).unwrap();
        println!(
            "Testing known: {:?} matches unknown {:?}",
            known_trace, trace
        );
        for (p, kp) in trace.iter().zip(known_trace) {
            assert_eq!(*p, kp);
        }
    }

    #[test]
    fn knn_singletons_on() {
        println!("2 nearest neighbors of 0.0 are 0.48 and 0.0");
        let writer = build_basic_tree();
        let reader = writer.reader();
        let zero_nbrs = reader.knn(&[0.1f32].as_ref(), 2).unwrap();
        println!("{:?}", zero_nbrs);
        assert!(zero_nbrs[0].1 == 4);
        assert!(zero_nbrs[1].1 == 2);
    }

    #[test]
    fn label_summary() {
        let data = vec![0.499, 0.49, 0.48, -0.49, 0.0];
        let labels = vec![0, 0, 0, 1, 1];

        let point_cloud = DefaultLabeledCloud::<L2>::new_simple(data, 1, labels);
        let builder = CoverTreeBuilder {
            scale_base: 2.0,
            leaf_cutoff: 1,
            min_res_index: -9,
            use_singletons: false,
            partition_type: PartitionType::Nearest,
            verbosity: 0,
            rng_seed: Some(0),
        };
        let mut tree = builder.build(Arc::new(point_cloud)).unwrap();
        tree.generate_summaries();
        let reader = tree.reader();

        for (_, layer) in reader.layers() {
            layer.for_each_node(|_, n| println!("{:?}", n.label_summary()));
        }

        let l = reader
            .get_node_label_summary(reader.root_address())
            .unwrap();
        assert_eq!(l.summary.items.len(), 2);
        assert_eq!(l.nones, 0);
        assert_eq!(l.errors, 0);
    }

    #[test]
    fn knn_singletons_off() {
        let data = vec![0.499, 0.49, 0.48, -0.49, 0.0];
        let labels = vec![0, 0, 0, 1, 1];

        let point_cloud = DefaultLabeledCloud::<L2>::new_simple(data, 1, labels);
        let builder = CoverTreeBuilder {
            scale_base: 2.0,
            leaf_cutoff: 1,
            min_res_index: -9,
            use_singletons: false,
            partition_type: PartitionType::Nearest,
            verbosity: 0,
            rng_seed: Some(0),
        };
        let tree = builder.build(Arc::new(point_cloud)).unwrap();
        let reader = tree.reader();

        println!("2 nearest neighbors of 0.1 are 0.48 and 0.0");
        let zero_nbrs = reader.knn(&[0.1f32].as_ref(), 2).unwrap();
        println!("{:?}", zero_nbrs);
        assert!(zero_nbrs[0].1 == 4);
        assert!(zero_nbrs[1].1 == 2);
    }

    #[test]
    fn test_save_load_tree() {
        let data = vec![0.499, 0.49, 0.48, -0.49, 0.0];
        let labels = vec![0, 0, 0, 1, 1];

        let point_cloud = Arc::new(DefaultLabeledCloud::<L2>::new_simple(data, 1, labels));
        let builder = CoverTreeBuilder {
            scale_base: 2.0,
            leaf_cutoff: 1,
            min_res_index: -9,
            use_singletons: false,
            partition_type: PartitionType::Nearest,
            verbosity: 0,
            rng_seed: Some(0),
        };
        let tree = builder.build(Arc::clone(&point_cloud)).unwrap();
        let reader = tree.reader();
        let proto = tree.save();

        assert_eq!(reader.layers.len(), proto.get_layers().len());

        for (layer, proto_layer) in reader.layers.iter().zip(proto.get_layers()) {
            assert_eq!(layer.len(), proto_layer.get_nodes().len());
        }

        let reconstructed_tree_writer =
            CoverTreeWriter::load(&proto, Arc::clone(&point_cloud)).unwrap();
        let reconstructed_tree = reconstructed_tree_writer.reader();

        assert_eq!(reader.layers.len(), reconstructed_tree.layers.len());
        for (layer, reconstructed_layer) in reader.layers.iter().zip(reconstructed_tree.layers) {
            assert_eq!(layer.len(), reconstructed_layer.len());

            layer.for_each_node(|pi, n| {
                reconstructed_layer
                    .get_node_and(*pi, |rn| {
                        assert_eq!(n.address(), rn.address());
                        assert_eq!(n.parent_address(), rn.parent_address());
                        assert_eq!(n.singletons(), rn.singletons());
                    })
                    .unwrap();
            })
        }
    }
}
