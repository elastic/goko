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

use crate::*;
use layer::*;
use node::*;
//use pointcloud::*;

use tree_file_format::*;
use std::sync::{atomic, Arc};

use crate::query_tools::KnnQueryHeap;
use errors::MalwareBrotResult;
use std::iter::Iterator;
use std::ops::Range;
use std::slice::Iter;

/// Container for the parameters governing the construction of the covertree
#[derive(Debug)]
pub struct CoverTreeParameters<M: Metric> {
    /// An atomic that tracks all nodes as they are created across all threads. 
    /// This may not reflect what your current reader can see.
    pub total_nodes: atomic::AtomicUsize,
    /// See paper or main description, governs the number of children of each node. Higher is more.
    pub scale_base: f32,
    /// If a node covers less than or equal to this number of points, it becomes a leaf.
    pub cutoff: usize,
    /// If a node has scale index less than or equal to this, it becomes a leaf
    pub resolution: i32,
    /// If you don't want singletons messing with your tree and want everything to be a node or a element of leaf node, make this true. 
    pub use_singletons: bool,
    /// Clustering is currently slow, avoid
    pub cluster_min: usize,
    /// The point cloud this tree references
    pub point_cloud: PointCloud<M>,
    /// This should be replaced by a logging solution
    pub verbosity: u32,
}

impl<M: Metric> CoverTreeParameters<M> {
    #[inline]
    pub(crate) fn internal_index(&self, scale_index: i32) -> usize {
        if scale_index < self.resolution {
            0
        } else {
            (scale_index - self.resolution + 1) as usize
        }
    }
}

/// Helper struct for iterating thru the reader's of the the layers. 
pub struct LayerIter<'a> {
    scales: Range<i32>,
    layers: Iter<'a, CoverLayerReader>,
}

impl<'a> Iterator for LayerIter<'a> {
    type Item = (i32, &'a CoverLayerReader);
    fn next(&mut self) -> Option<Self::Item> {
        match (self.scales.next(), self.layers.next()) {
            (Some(a), Some(b)) => Some((a, b)),
            _ => None,
        }
    }
}

/// # Cover Tree Reader Head
///
/// You can clone the reader head, though this is a relatively expensive operation and should not be performed lightly.
/// 
/// All queries of the covertree should go through a reader head. This includes queries you are doing to modify the tree.
/// There are no thread locks anywhere in the code below the reader head, so it's fast. 
///
/// The data structure is just a list of `CoverLayerReader`s, the parameter's object and the root address. Copies are relatively
/// expensive as each `CoverLayerReader` contains several Arcs that need to be cloned.
pub struct CoverTreeReader<M: Metric> {
    parameters: Arc<CoverTreeParameters<M>>,
    layers: Vec<CoverLayerReader>,
    root_address: NodeAddress,
}

impl<M: Metric> CoverTreeReader<M> {
    /// A reference to the point cloud the tree was built on.
    pub fn point_cloud(&self) -> &PointCloud<M> {
        &self.parameters.point_cloud
    }

    /// Returns a borrowed reader for a cover layer. 
    /// 
    pub fn layer(&self, scale_index: i32) -> &CoverLayerReader {
        &self.layers[self.parameters.internal_index(scale_index)]
    }

    /// simple helper to get the scale from the scale index and the scale base, this is just `b^i`
    pub fn scale(&self, scale_index: i32) -> f32 {
        self.parameters.scale_base.powi(scale_index)
    }

    /// Read only access to the internals of a node. 
    pub fn get_node_and<F, T>(&self, node_address: (i32, PointIndex), f: F) -> Option<T>
    where
        F: FnOnce(&CoverNode) -> T,
    {
        self.layers[self.parameters.internal_index(node_address.0)]
            .get_node_and(&node_address.1, |n| f(n))
    }

    /// The root of the tree. Pass this to `get_node_and` to get the root node's content and start a traversal of the tree.
    pub fn root_address(&self) -> NodeAddress {
        self.root_address
    }

    /// 
    pub fn layers<'a>(&'a self) -> LayerIter<'a> {
        LayerIter {
            scales: (self.layers.len() as i32 - self.parameters.resolution)
                ..(self.parameters.resolution),
            layers: self.layers.iter(),
        }
    }

    /// If you want to build a new tree with shared parameters, this is helpful.
    pub fn parameters(&self) -> &Arc<CoverTreeParameters<M>> {
        &self.parameters
    }

    /// This is the total number of nodes in the tree. This queries each layer, so it's not a simple return int.
    pub fn node_count(&self) -> usize {
        self.layers().fold(0,|a,(_si,l)| a+l.node_count())
    }

    /// Returns the scale index range. It starts at the minimum resolution and ends at the top. You can reverse this for the correct order.
    pub fn scale_range(&self) -> Range<i32> {
        (self.parameters.resolution)..(self.parameters.resolution - 1 + self.layers.len() as i32)
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
    pub fn knn(&self,point:&[f32],k:usize) -> MalwareBrotResult<Vec<(f32,PointIndex)>> {
        let mut query_heap = KnnQueryHeap::new(k, self.parameters.scale_base);

        let root_center = self.parameters.point_cloud.get_point(self.root_address.1)?;
        let dist_to_root = M::dense(root_center,point);
        query_heap.push_nodes(&[self.root_address],&[dist_to_root],None);
        self.greedy_knn_nodes(&point,&mut query_heap);

        while let Some((_dist,address)) = query_heap.closest_unvisited_singleton_covering_address() {
            self.get_node_and(address, |n| n.singleton_knn(point,&self.parameters.point_cloud,&mut query_heap));
            self.greedy_knn_nodes(&point,&mut query_heap);
        }
        
        Ok(query_heap.unpack())
    }
    
    fn greedy_knn_nodes(&self, point: &[f32], query_heap: &mut KnnQueryHeap) {
        loop {
            if let Some((dist, nearest_address)) = query_heap.closest_unvisited_child_covering_address() {
                if self
                    .get_node_and(nearest_address, |n| n.is_leaf())
                    .unwrap_or(true)
                {
                    break;
                } else {
                    self.get_node_and(nearest_address, |n| {
                        n.child_knn(Some(dist), point, &self.parameters.point_cloud, query_heap)
                    });
                }
            } else {
                break;
            }
        }
    }

    /// Checks that there are no node addresses in the child list of any node that don't reference a node in the tree. 
    /// Please calmly panic if there are, the tree is very invalid.
    pub(crate) fn no_dangling_refs(&self) -> bool {
        let mut refs_to_check = vec![self.root_address];
        while let Some(node_addr) = refs_to_check.pop() {
            if self
                .get_node_and(node_addr, |n| {
                    match n.children() {
                        None => {}
                        Some((nested_scale, other_children)) => {
                            refs_to_check.push((nested_scale, node_addr.1));
                            refs_to_check.extend(&other_children[..]);
                        }
                    };
                })
                .is_none()
            {
                return false;
            }
        }
        true
    }

    fn cluster_children(
        &self,
        si: i32,
        pis: &[PointIndex],
    ) -> MalwareBrotResult<Vec<(usize, i32, Vec<PointIndex>)>> {
        //println!("\tClustering children of {:?}", (si,pis));
        let mut children_addresses = Vec::new();
        for pi in pis {
            self.layer(si).get_node_children_and(&pi, |na, nas| {
                children_addresses.push(na);
                children_addresses.extend(nas);
            });
        }
        println!(
            "\t There are {:?} total children, with an average spawn of {}",
            children_addresses.len(),
            (children_addresses.len() as f32) / (pis.len() as f32)
        );

        let mut scale_indexes: Vec<i32> = children_addresses.iter().map(|(si, _pi)| *si).collect();
        ////println!("\t Child addresses are {:?}", children_addresses);
        scale_indexes.dedup();
        let mut clusters = Vec::new();
        for child_si in scale_indexes {
            let unclustered: Vec<PointIndex> = children_addresses
                .iter()
                .filter(|(osi, _pi)| osi == &child_si)
                .map(|(_si, pi)| *pi)
                .collect();
            //println!("\t\t[{}] Clustering on {:?} points", child_si, unclustered.len());
            let mut components = self
                .layer(child_si)
                .get_components(unclustered, &self.parameters.point_cloud)?;
            //println!("\t\t[{}] Obtained {:?} clusters", child_si, components);

            while let Some((id, component)) = components.pop() {
                clusters.push((id, child_si, component));
            }
        }
        Ok(clusters)
    }
}

/// 
pub struct CoverTreeWriter<M: Metric> {
    pub(crate) parameters: Arc<CoverTreeParameters<M>>,
    pub(crate) layers: Vec<CoverLayerWriter>,
    pub(crate) root_address: NodeAddress,
}

impl<M: Metric> CoverTreeWriter<M> {
    #[doc(hidden)]
    pub fn cluster(&mut self) -> MalwareBrotResult<()> {
        let reader = self.reader();
        let mut pending_clusters = vec![(0, self.root_address.0, vec![self.root_address.1])];
        while let Some((i, si, pis)) = pending_clusters.pop() {
            println!(
                "Clustering children of cluster_id:{:?}, len: {}",
                (i, si),
                pis.len()
            );
            let mut children_clusters = reader.cluster_children(si, &pis[..])?;
            let mut children_ids = Vec::new();
            while let Some((id, nsi, npis)) = children_clusters.pop() {
                pending_clusters.push((id, nsi, npis));
                children_ids.push((nsi, id));
            }
            unsafe {self.layer(si).insert_cluster(
                i,
                CoverCluster {
                    indexes: pis,
                    children_ids,
                },
            );}
        }
        Ok(())
    }

    /// Provides a reference to a `CoverLayerWriter`. Do not use, unless you're going to leave the tree in a *valid* state.
    pub(crate) unsafe fn layer(&mut self, scale_index: i32) -> &mut CoverLayerWriter {
        &mut self.layers[self.parameters.internal_index(scale_index)]
    }

    /// Creates a reader for queries.
    pub fn reader(&self) -> CoverTreeReader<M> {
        CoverTreeReader {
            parameters: Arc::clone(&self.parameters),
            layers: self.layers.iter().map(|l| l.reader()).collect(),
            root_address: self.root_address,
        }
    }

    pub(crate) unsafe fn insert_raw(
        &mut self,
        scale_index: i32,
        point_index: PointIndex,
        node: CoverNode,
    ) {
        self.layers[self.parameters.internal_index(scale_index)].insert_raw(point_index, node);
    }

    /// Loads a tree from a protobuf. There's a `load_tree` in `utils` that handles loading from a path to a protobuf file.
    pub fn load(
        cover_proto: &CoreProto,
        point_cloud: PointCloud<M>,
    ) -> MalwareBrotResult<CoverTreeWriter<M>> {
        let parameters = Arc::new(CoverTreeParameters {
            total_nodes: atomic::AtomicUsize::new(0),
            use_singletons: cover_proto.use_singletons,
            scale_base: cover_proto.scale_base as f32,
            cutoff: cover_proto.cutoff as usize,
            resolution: cover_proto.resolution as i32,
            cluster_min: 5,
            point_cloud,
            verbosity: 2,
        });
        let root_address = (cover_proto.get_root_scale(), cover_proto.get_root_index());
        let layers = cover_proto
            .get_layers()
            .par_iter()
            .map(|l| CoverLayerWriter::load(l))
            .collect();

        Ok(CoverTreeWriter {
            parameters,
            layers,
            root_address,
        })
    }

    /// Encodes the tree into a protobuf. See `utils::save_tree` for saving to a file on disk.
    pub fn save(&self) -> CoreProto {
        let mut cover_proto = CoreProto::new();
        cover_proto.set_scale_base(self.parameters.scale_base);
        cover_proto.set_cutoff(self.parameters.cutoff as u64);
        cover_proto.set_resolution(self.parameters.resolution);
        cover_proto.set_use_singletons(self.parameters.use_singletons);
        cover_proto.set_dim(self.parameters.point_cloud.dim() as u64);
        cover_proto.set_count(self.parameters.point_cloud.len() as u64);
        cover_proto.set_root_scale(self.root_address.0);
        cover_proto.set_root_index(self.root_address.1);
        cover_proto.set_layers(self.layers.iter().map(|l| l.save()).collect());
        cover_proto
    }

    /// Swaps the maps on each layer so that any `CoverTreeReaders` see the updated tree. 
    /// Only call once you have a valid tree.
    pub fn refresh(&mut self) {
        self.layers.par_iter_mut().for_each(|l| l.refresh());
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::utils::cover_tree_from_yaml;
    use std::path::Path;

    pub(crate) fn build_mnist_tree() -> CoverTreeWriter<L2> {
        let file_name = "data/mnist_complex.yml";
        let path = Path::new(file_name);
        if !path.exists() {
            panic!(file_name.to_owned() + &" does not exist".to_string());
        }
        cover_tree_from_yaml(&path).unwrap()
    }

    #[test]
    fn greedy_knn_nodes() {
        let data = vec![0.499, 0.49, 0.48, -0.49, 0.0];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0];

        let point_cloud =
            PointCloud::<L2>::simple_from_ram(Box::from(data), 1, Box::from(labels), 1).unwrap();
        let builder = CoverTreeBuilder {
            scale_base: 2.0,
            cutoff: 1,
            resolution: -9,
            use_singletons: false,
            cluster_min: 5,
            verbosity: 0,
        };
        let tree = builder.build(point_cloud).unwrap();
        let reader = tree.reader();

        let point = [-0.5];

        let mut query_heap = KnnQueryHeap::new(5, reader.parameters.scale_base);
        let dist_to_root = reader
            .parameters
            .point_cloud
            .distances_to_point(&point, &[reader.root_address().1])
            .unwrap()[0];
        query_heap.push_nodes(&[reader.root_address()], &[dist_to_root], None);

        assert_eq!(
            reader.root_address(),
            query_heap.closest_unvisited_child_covering_address().unwrap().1
        );

        reader.greedy_knn_nodes(&point,&mut query_heap);
        println!("{:#?}", query_heap);
        println!("{:#?}",query_heap.closest_unvisited_child_covering_address());
    }
    
    #[test]
    fn knn_singletons_on() {
        let data = vec![0.499, 0.49, 0.48, -0.49, 0.0];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0];

        let point_cloud =
            PointCloud::<L2>::simple_from_ram(Box::from(data), 1, Box::from(labels), 1).unwrap();
        let builder = CoverTreeBuilder {
            scale_base: 2.0,
            cutoff: 1,
            resolution: -9,
            use_singletons: true,
            cluster_min: 5,
            verbosity: 0,
        };
        let tree = builder.build(point_cloud).unwrap();
        let reader = tree.reader();

        println!("2 nearest neighbors of 0.0 are 0.48 and 0.0");
        let zero_nbrs = reader.knn(&[0.1],2).unwrap();
        println!("{:?}", zero_nbrs);
        assert!(zero_nbrs[0].1 == 4);
        assert!(zero_nbrs[1].1 == 2);
    }

    #[test]
    fn knn_singletons_off() {
        let data = vec![0.499, 0.49, 0.48, -0.49, 0.0];
        let labels = vec![0.0, 0.0, 0.0, 1.0, 1.0];

        let point_cloud =
            PointCloud::<L2>::simple_from_ram(Box::from(data), 1, Box::from(labels), 1).unwrap();
        let builder = CoverTreeBuilder {
            scale_base: 2.0,
            cutoff: 1,
            resolution: -9,
            use_singletons: false,
            cluster_min: 5,
            verbosity: 0,
        };
        let tree = builder.build(point_cloud).unwrap();
        let reader = tree.reader();

        println!("2 nearest neighbors of 0.1 are 0.48 and 0.0");
        let zero_nbrs = reader.knn(&[0.1],2).unwrap();
        println!("{:?}", zero_nbrs);
        assert!(zero_nbrs[0].1 == 4);
        assert!(zero_nbrs[1].1 == 2);
    }
}
