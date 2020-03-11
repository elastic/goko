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

//! # The Layers
//! This is the actual pair of hashmaps mentioned in `tree`, the `evmap` imported here is a modification of
//! Jon Gjengset's. It should probably be modified more and pulled more into this library as this is the main bottleneck
//! of the library.
//!
//! Each layer consists of a pair of hash-maps where the values are nodes and the keys are the index of the center
//! point of the node. This uniquely identifies each node on a layer and is a meaningful index pattern.
//!
//! Writes to the tree are written to each layer and then each layer is refreshed. You should refrain from refreshing
//! single layers and try to handle all write operations as a tree level function.
//!
//! There is also an experimental pair of cluster hashmaps, which need to be replaced by a data structure that
//! respects and represents the nerve more.

use crate::evmap::monomap::{MonoReadHandle, MonoWriteHandle};
use pointcloud::*;

//use rayon;
use super::*;
use node::*;
use pointcloud::utils::AdjMatrix;
use std::iter::FromIterator;
use std::sync::{atomic, Arc};
use tree_file_format::*;

/// Actual reader, primarily contains a read head to the hash-map.
/// This also contains a reference to the scale_index so that it is easy to save and load. It is largely redundant,
/// but helps with unit tests.
pub struct CoverLayerReader<M: Metric> {
    scale_index: i32,
    node_reader: MonoReadHandle<PointIndex, CoverNode<M>>,
    cluster_reader: MonoReadHandle<usize, CoverCluster>,
    cluster_index: Arc<atomic::AtomicUsize>,
}

impl<M: Metric> CoverLayerReader<M> {
    /// Read only access to a single node.
    pub fn get_node_and<F, T>(&self, pi: &PointIndex, f: F) -> Option<T>
    where
        F: FnOnce(&CoverNode<M>) -> T,
    {
        self.node_reader.get_and(pi, |n| f(n))
    }

    /// Read only access to all nodes.
    pub fn for_each_node<F>(&self, f: F)
    where
        F: FnMut(&PointIndex, &CoverNode<M>),
    {
        self.node_reader.for_each(f)
    }

    /// Maps all nodes on the layer, useful for collecting statistics.
    pub fn map_nodes<Map, Target, Collector>(&self, f: Map) -> Collector
    where
        Map: FnMut(&PointIndex, &CoverNode<M>) -> Target,
        Collector: FromIterator<Target>,
    {
        self.node_reader.map_into(f)
    }

    /// Grabs all children indexes and allows you to query against them. Usually used at the tree level so that you
    /// can access the child nodes as they are not on this layer.
    pub fn get_node_children_and<F, T>(&self, pi: &PointIndex, f: F) -> Option<T>
    where
        F: FnOnce(NodeAddress, &[NodeAddress]) -> T,
    {
        self.node_reader
            .get_and(pi, |n| n.children().map(|(si, c)| f((si, *pi), c)))
            .flatten()
    }

    #[doc(hidden)]
    pub fn get_cluster_and<F, T>(&self, pi: &usize, f: F) -> Option<T>
    where
        F: FnOnce(&CoverCluster) -> T,
    {
        self.cluster_reader.get_and(pi, |n| f(n))
    }

    #[doc(hidden)]
    pub fn cluster_count(&self) -> usize {
        self.cluster_reader.len()
    }

    /// Total number of nodes on this layer
    pub fn node_count(&self) -> usize {
        self.node_reader.len()
    }

    /// Read only accessor for the scale index.
    pub fn scale_index(&self) -> i32 {
        self.scale_index
    }

    /// Clones the reader, expensive!
    pub fn reader(&self) -> CoverLayerReader<M> {
        CoverLayerReader {
            scale_index: self.scale_index,
            node_reader: self.node_reader.factory().handle(),
            cluster_reader: self.cluster_reader.factory().handle(),
            cluster_index: Arc::clone(&self.cluster_index),
        }
    }

    #[doc(hidden)]
    pub fn get_components(
        &self,
        mut unclustered: Vec<PointIndex>,
        point_cloud: &PointCloud<M>,
    ) -> GrandmaResult<Vec<(usize, Vec<PointIndex>)>> {
        if unclustered.len() == 1 {
            //println!("\t\t Singleton with {:?}", unclustered);
            let id = self.cluster_index.fetch_add(1, atomic::Ordering::SeqCst);
            Ok(vec![(id, unclustered)])
        } else {
            let mut new_clusters = Vec::new();
            let adj = point_cloud.adj(&unclustered)?;
            while let Some(i) = unclustered.pop() {
                //println!("\t\t\t Building cluster around {:?}", i);
                let id = self.cluster_index.fetch_add(1, atomic::Ordering::SeqCst);
                let indexes = self.get_components_rec(&i, &mut unclustered, &adj);
                new_clusters.push((id, indexes));
            }
            Ok(new_clusters)
        }
    }

    fn get_components_rec(
        &self,
        center: &PointIndex,
        unclustered: &mut Vec<PointIndex>,
        adj: &AdjMatrix,
    ) -> Vec<PointIndex> {
        let mut unvisited = vec![*center];
        let mut component = vec![*center];

        while let Some(i) = unvisited.pop() {
            //println!("\t\t\t\t getting radius of {}",i);
            let i_radius = self.node_reader.get_and(&i, |n| n.radius()).unwrap();
            unclustered.retain(|j| {
                //println!("\t\t\t\t\t getting radius of {}",j);
                let j_radius = self.node_reader.get_and(&j, |n| n.radius()).unwrap();
                if *adj.get(i, *j).unwrap() < i_radius + j_radius {
                    //println!("\t\t\t\t\t {} and {} have radii {} and {} and are dist {}",i,j,i_radius,j_radius, *adj.get(i, *j).unwrap());
                    component.push(*j);
                    unvisited.push(*j);
                    //println!("\t\t\t\t\t component: {:?}", component);
                    //println!("\t\t\t\t\t unvisited: {:?}", unvisited);
                    false
                } else {
                    true
                }
            });
        }
        component
    }

    /// Checks if this is a true cover tree layer.
    pub fn brute_check_seperability(
        &self,
        distances: &[f32],
        cluster_list: &[PointIndex],
        existing_cluster: &[PointIndex],
    ) -> bool {
        for i in 0..cluster_list.len() {
            let i_r = self
                .node_reader
                .get_and(&cluster_list[i], |n| n.radius())
                .unwrap();
            for j in 0..existing_cluster.len() {
                let j_r = self
                    .node_reader
                    .get_and(&existing_cluster[j], |n| n.radius())
                    .unwrap();
                if distances[i * existing_cluster.len() + j] < i_r + j_r {
                    return true;
                }
            }
        }
        false
    }
}

#[doc(hidden)]
#[derive(Clone)]
pub struct CoverCluster {
    pub indexes: Vec<PointIndex>,
    pub children_ids: Vec<ClusterAddress>,
}

/// Primarily contains the node writer head, but also has the cluster writer head and the index head.
pub(crate) struct CoverLayerWriter<M: Metric> {
    scale_index: i32,
    node_writer: MonoWriteHandle<PointIndex, CoverNode<M>>,
    cluster_writer: MonoWriteHandle<usize, CoverCluster>,
    cluster_index: Arc<atomic::AtomicUsize>,
}

impl<M: Metric> CoverLayerWriter<M> {
    /// Creates a reader head. Only way to get one from a newly created layer.
    pub(crate) fn reader(&self) -> CoverLayerReader<M> {
        CoverLayerReader {
            scale_index: self.scale_index,
            node_reader: self.node_writer.factory().handle(),
            cluster_reader: self.cluster_writer.factory().handle(),
            cluster_index: Arc::clone(&self.cluster_index),
        }
    }

    /// Constructs the object. To construct a reader call `reader`.
    pub(crate) fn new(scale_index: i32) -> CoverLayerWriter<M> {
        let (_node_reader, node_writer) = evmap::monomap::new();
        let (_cluster_reader, cluster_writer) = evmap::monomap::new::<usize, CoverCluster>();
        CoverLayerWriter {
            scale_index,
            cluster_writer,
            node_writer,
            cluster_index: Arc::new(atomic::AtomicUsize::new(0)),
        }
    }

    pub(crate) unsafe fn update_node<F>(&mut self, pi: PointIndex, update_fn: F)
    where
        F: Fn(&mut CoverNode<M>) + 'static + Send + Sync,
    {
        self.node_writer.update(pi, update_fn);
    }

    #[doc(hidden)]
    pub(crate) fn insert_cluster(&mut self, index: usize, cluster: CoverCluster) {
        self.cluster_writer.insert(index, cluster);
    }

    pub(crate) fn load(layer_proto: &LayerProto) -> CoverLayerWriter<M> {
        let scale_index = layer_proto.get_scale_index();
        let (_node_reader, mut node_writer) = evmap::monomap::new();
        let (_cluster_reader, cluster_writer) = evmap::monomap::new::<usize, CoverCluster>();
        for node_proto in layer_proto.get_nodes() {
            let index = node_proto.get_center_index() as PointIndex;
            let node = CoverNode::load(scale_index, node_proto);
            node_writer.insert(index, node);
        }
        node_writer.refresh();
        CoverLayerWriter {
            scale_index,
            cluster_writer,
            node_writer,
            cluster_index: Arc::new(atomic::AtomicUsize::new(0)),
        }
    }

    pub(crate) fn save(&self) -> LayerProto {
        let mut layer_proto = LayerProto::new();
        let mut node_protos = layer_proto.take_nodes();
        self.node_writer.for_each(|_pi, node| {
            node_protos.push(node.save());
        });
        layer_proto.set_nodes(node_protos);
        layer_proto.set_scale_index(self.scale_index);
        layer_proto
    }

    pub(crate) fn insert_raw(&mut self, index: PointIndex, node: CoverNode<M>) {
        self.node_writer.insert(index, node);
    }

    pub(crate) fn refresh(&mut self) {
        self.node_writer.refresh();
        self.cluster_writer.refresh();
    }
}
