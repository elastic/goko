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

use crate::evmap::{MonoReadHandle, MonoWriteHandle};
use pointcloud::*;

//use rayon;
use crate::*;
use super::node::*;
use std::iter::FromIterator;
use crate::tree_file_format::*;

/// Actual reader, primarily contains a read head to the hash-map.
/// This also contains a reference to the scale_index so that it is easy to save and load. It is largely redundant,
/// but helps with unit tests.
pub struct CoverLayerReader<D: PointCloud> {
    scale_index: i32,
    node_reader: MonoReadHandle<PointIndex, CoverNode<D>>,
}

impl<D: PointCloud> Clone for CoverLayerReader<D> {
    fn clone(&self) -> CoverLayerReader<D> {
        CoverLayerReader {
            scale_index: self.scale_index,
            node_reader: self.node_reader.clone(),
        }
    }
}

impl<D: PointCloud> CoverLayerReader<D> {
    /// Read only access to a single node.
    pub fn get_node_and<F, T>(&self, pi: PointIndex, f: F) -> Option<T>
    where
        F: FnOnce(&CoverNode<D>) -> T,
    {
        self.node_reader.get_and(&pi, |n| f(n))
    }

    /// Reads the contents of a plugin, due to the nature of the plugin map we have to access it with a
    /// closure.
    pub fn get_node_plugin_and<T: Send + Sync + 'static, F, S>(
        &self,
        center_index: PointIndex,
        transform_fn: F,
    ) -> Option<S>
    where
        F: FnOnce(&T) -> S,
    {
        self.get_node_and(center_index, |n| n.get_plugin_and(transform_fn))
            .flatten()
    }

    /// Read only access to all nodes.
    pub fn for_each_node<F>(&self, f: F)
    where
        F: FnMut(&PointIndex, &CoverNode<D>),
    {
        self.node_reader.for_each(f)
    }

    /// Maps all nodes on the layer, useful for collecting statistics.
    pub fn map_nodes<Map, Target, Collector>(&self, f: Map) -> Collector
    where
        Map: FnMut(&PointIndex, &CoverNode<D>) -> Target,
        Collector: FromIterator<Target>,
    {
        self.node_reader.map_into(f)
    }

    /// Grabs all children indexes and allows you to query against them. Usually used at the tree level so that you
    /// can access the child nodes as they are not on this layer.
    pub fn get_node_children_and<F, T>(&self, pi: PointIndex, f: F) -> Option<T>
    where
        F: FnOnce(NodeAddress, &[NodeAddress]) -> T,
    {
        self.node_reader
            .get_and(&pi, |n| n.children().map(|(si, c)| f((si, pi), c)))
            .flatten()
    }

    /// Grabs all children indexes and allows you to query against them. Usually used at the tree level so that you
    /// can access the child nodes as they are not on this layer.
    pub fn node_center_indexes(&self) -> Vec<PointIndex> {
        self.node_reader.map_into(|pi, _| *pi)
    }

    /// Total number of nodes on this layer
    pub fn len(&self) -> usize {
        self.node_reader.len()
    }

    /// Total number of nodes on this layer
    pub fn is_empty(&self) -> bool {
        self.node_reader.is_empty()
    }

    /// Read only accessor for the scale index.
    pub fn scale_index(&self) -> i32 {
        self.scale_index
    }

    /// Clones the reader, expensive!
    pub fn reader(&self) -> CoverLayerReader<D> {
        CoverLayerReader {
            scale_index: self.scale_index,
            node_reader: self.node_reader.factory().handle(),
        }
    }
}

/// Primarily contains the node writer head, but also has the cluster writer head and the index head.
pub struct CoverLayerWriter<D: PointCloud> {
    scale_index: i32,
    node_writer: MonoWriteHandle<PointIndex, CoverNode<D>>,
}

impl<D: PointCloud> CoverLayerWriter<D> {
    /// Creates a reader head. Only way to get one from a newly created layer.
    pub(crate) fn reader(&self) -> CoverLayerReader<D> {
        CoverLayerReader {
            scale_index: self.scale_index,
            node_reader: self.node_writer.factory().handle(),
        }
    }

    /// Constructs the object. To construct a reader call `reader`.
    pub(crate) fn new(scale_index: i32) -> CoverLayerWriter<D> {
        let (_node_reader, node_writer) = evmap::monomap::new();
        CoverLayerWriter {
            scale_index,
            node_writer,
        }
    }

    pub(crate) unsafe fn update_node<F>(&mut self, pi: PointIndex, update_fn: F)
    where
        F: Fn(&mut CoverNode<D>) + 'static + Send + Sync,
    {
        self.node_writer.update(pi, update_fn);
    }

    pub(crate) fn load(layer_proto: &LayerProto) -> CoverLayerWriter<D> {
        let scale_index = layer_proto.get_scale_index();
        let (_node_reader, mut node_writer) = evmap::monomap::new();
        for node_proto in layer_proto.get_nodes() {
            let index = node_proto.get_center_index() as PointIndex;
            let node = CoverNode::load(node_proto);
            node_writer.insert(index, node);
        }
        node_writer.refresh();
        node_writer.refresh();
        CoverLayerWriter {
            scale_index,
            node_writer,
        }
    }

    /// Read only accessor for the scale index.
    pub(crate) fn scale_index(&self) -> i32 {
        self.scale_index
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

    pub(crate) fn insert_raw(&mut self, index: PointIndex, node: CoverNode<D>) {
        self.node_writer.insert(index, node);
    }

    pub(crate) fn refresh(&mut self) {
        self.node_writer.refresh();
    }
}
