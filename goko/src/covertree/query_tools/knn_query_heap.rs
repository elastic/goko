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

//! Tools and data structures for assisting cover tree queries.

use crate::{NodeAddress, NodeAddressBase};
use std::collections::{BinaryHeap, HashMap, HashSet};
use std::f32;

use super::*;

use super::query_items::{QueryAddress, QuerySingleton};

/// The heaps for doing a fairly efficient KNN query. There are 3 heaps, the child min-heap, singleton min-heap, and distance max-heap.
/// The distance heap is for the output KNN, each node or point that's pushed onto the heap is pushed onto this distance heap.
/// If the heap grows past K it's popped off. This provides an estimate for the distance to the furthest nearest neighbor out of the `k`.
///
/// The child and singleton heaps are for nodes only. The names are a bit of a misnomer, the child heap is for nodes where we haven't checked their
/// children yet, and the singleton heap is for those nodes where we haven't checked their singletons. Next to these is a hashmap that records the
/// minimum distance a point could have to a point covered by that node. Togther with the current max distance (from the distance max-heap)
/// K this can help with the KNN query.
///
/// To help with double inserts (easy due to a node's central point's index being repeated througout the tree), we also have a HashSet of visited points.
/// We reject a node insert if it's central point index is in this hashset.
///
#[derive(Debug)]
pub struct KnnQueryHeap {
    child_heap: BinaryHeap<QueryAddress>,
    singleton_heap: BinaryHeap<QueryAddress>,

    known_indexes: HashSet<usize>,
    est_min_dist: HashMap<NodeAddress, f32>,
    dist_heap: BinaryHeap<QuerySingleton>,
    k: usize,
    scale_base: f32,
}

impl RoutingQueryHeap for KnnQueryHeap {
    /// Shove a bunch of nodes onto the heap. Optionally, if you pass a parent node it updates the distance to that parent node.
    fn push_nodes(
        &mut self,
        indexes: &[NodeAddress],
        dists: &[f32],
        parent_address: Option<NodeAddress>,
    ) {
        let mut max_dist = self.max_dist();
        let mut parent_est_dist_update = 0.0;
        for (na, d) in indexes.iter().zip(dists) {
            let pi = na.point_index();
            let emd = (d - self.scale_base.powi(na.scale_index())).max(0.0);
            parent_est_dist_update = emd.max(parent_est_dist_update);
            if emd < max_dist {
                self.child_heap.push(QueryAddress {
                    address: *na,
                    dist_to_center: *d,
                    min_dist: emd,
                });
            }
            if !self.known_indexes.contains(&pi) {
                self.known_indexes.insert(pi);
                match self.dist_heap.peek() {
                    Some(my_dist) => {
                        if !(my_dist.dist < *d && self.dist_heap.len() >= self.k) {
                            self.dist_heap.push(QuerySingleton::new(pi, *d));
                        }
                    }
                    None => self.dist_heap.push(QuerySingleton::new(pi, *d)),
                };
            }
            while self.dist_heap.len() > self.k {
                self.dist_heap.pop();
                max_dist = self.max_dist();
            }
        }

        if let Some(a) = parent_address {
            self.increase_estimated_distance(a, parent_est_dist_update);
        }
    }
}

impl SingletonQueryHeap for KnnQueryHeap {
    /// Shove a bunch of single points onto the heap
    fn push_outliers(&mut self, indexes: &[usize], dists: &[f32]) {
        for (i, d) in indexes.iter().zip(dists) {
            if !self.known_indexes.contains(i) {
                self.known_indexes.insert(*i);
                match self.dist_heap.peek() {
                    Some(my_dist) => {
                        if !(my_dist.dist < *d && self.dist_heap.len() >= self.k) {
                            self.dist_heap.push(QuerySingleton::new(*i, *d));
                        }
                    }
                    None => self.dist_heap.push(QuerySingleton::new(*i, *d)),
                };
                while self.dist_heap.len() > self.k {
                    self.dist_heap.pop();
                }
            }
        }
    }
}

impl KnnQueryHeap {
    /// Creates a new KNN heap. The K is obvious, but the `scale_base` is for the
    /// minimum distance from our query point to potential covered points of a node.
    pub fn new(k: usize, scale_base: f32) -> KnnQueryHeap {
        KnnQueryHeap {
            child_heap: BinaryHeap::new(),
            singleton_heap: BinaryHeap::new(),
            est_min_dist: HashMap::new(),
            dist_heap: BinaryHeap::new(),
            known_indexes: HashSet::new(),
            k,
            scale_base,
        }
    }

    /// Finds the closest node who could have a child node at least the current kth furthest distance away from the query point.
    /// This pops that node and pushes it onto the singleton heap.
    pub fn closest_unvisited_child_covering_address(&mut self) -> Option<(f32, NodeAddress)> {
        while let Some(mut node_to_visit) = self.child_heap.pop() {
            if let Some(min_dist_update) = self.est_min_dist.remove(&node_to_visit.address) {
                if min_dist_update > node_to_visit.min_dist {
                    node_to_visit.min_dist = min_dist_update;
                    self.child_heap.push(node_to_visit);
                } else {
                    self.singleton_heap.push(node_to_visit);
                    return Some((node_to_visit.dist_to_center, node_to_visit.address));
                }
            } else {
                self.singleton_heap.push(node_to_visit);
                return Some((node_to_visit.dist_to_center, node_to_visit.address));
            }
        }
        None
    }

    /// Finds the closest node who could have a singleton at least the current kth furthest distance away from the query point.
    /// This pops the node and sends it to oblivion.
    pub fn closest_unvisited_singleton_covering_address(&mut self) -> Option<(f32, NodeAddress)> {
        while let Some(mut node_to_visit) = self.singleton_heap.pop() {
            if let Some(min_dist_update) = self.est_min_dist.remove(&node_to_visit.address) {
                if min_dist_update > node_to_visit.min_dist {
                    node_to_visit.min_dist = min_dist_update;
                    self.singleton_heap.push(node_to_visit);
                } else {
                    return Some((node_to_visit.dist_to_center, node_to_visit.address));
                }
            } else {
                return Some((node_to_visit.dist_to_center, node_to_visit.address));
            }
        }
        None
    }

    /// The current number of points on the distance heap
    pub fn len(&self) -> usize {
        self.dist_heap.len()
    }

    /// The current number of points on the distance heap
    pub fn is_empty(&self) -> bool {
        self.dist_heap.is_empty()
    }

    /// The current number of points still on the
    pub fn node_len(&self) -> usize {
        self.child_heap.len() + self.singleton_heap.len()
    }

    /// The current maximum distance to the query point. If the distance heap isn't full it returns the maximum float value.
    pub fn max_dist(&self) -> f32 {
        if self.len() < self.k {
            std::f32::MAX
        } else {
            self.dist_heap.peek().map(|x| x.dist).unwrap_or(f32::MAX)
        }
    }

    /// Unpacks the distance heap. This consumes the query heap.
    pub fn unpack(mut self) -> Vec<(f32, usize)> {
        let mut result = Vec::with_capacity(self.k);
        while let Some(el) = self.dist_heap.pop() {
            result.push((el.dist, el.index));
        }
        result.iter().rev().cloned().collect()
    }

    /// This allows you to update the minimum distance to the parent of a node, or it's siblings.
    /// If you are well within the radius of coverage of a node, this allows you to remove the parent or sibling from the
    ///  `closest_unvisited_child_covering_address` and `closest_unvisited_singleton_covering_address` queries.
    pub fn increase_estimated_distance(&mut self, address: NodeAddress, new_estimate: f32) {
        let d = self.est_min_dist.entry(address).or_insert(0.0);
        if *d < new_estimate {
            *d = new_estimate;
        }
    }
}

//Tested in the node file too
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    #[test]
    fn unpacking_has_correct_order() {
        let mut heap = KnnQueryHeap::new(4, 2.0);
        heap.push_outliers(&[2, 4, 6, 8], &[0.2, 0.4, 0.6, 0.8]);
        heap.push_nodes(
            &[(0, 1), (0, 3), (1, 5), (1, 7)],
            &[0.1, 0.3, 0.5, 0.7],
            None,
        );
        let unpack = heap.unpack();

        for i in 1..5 {
            assert!(unpack[i - 1].1 == i);
        }
    }

    pub fn clone_unvisited_nodes(heap: &KnnQueryHeap) -> Vec<(f32, NodeAddress)> {
        let mut all_nodes: Vec<QueryAddress> = heap.child_heap.iter().cloned().collect();
        all_nodes.extend(heap.singleton_heap.iter().cloned());

        all_nodes.sort();
        all_nodes.iter().map(|n| (n.min_dist, n.address)).collect()
    }
    /*
        #[test]
        fn level_grab_is_correct() {
            let mut heap = KnnQueryHeap::new_scale_aware(4, 2.0);
            heap.push_outliers(&[2, 4, 6, 8], &[0.2, 0.4, 0.6, 0.8]);
            heap.push_nodes(
                &[(-4, 1), (0, 3), (0, 5), (-4, 7), (0, 8)],
                &[0.1, 0.3, 0.5, 0.7, 1.1],
                None,
            );
            println!("{:#?}", heap);
            let layer_one = heap.layer_estimated_distances(-4);
            println!("Layer -4: {:?}", layer_one);
            assert!(layer_one.len() == 1);
            assert!(layer_one[0].1 == 1);
            assert_approx_eq!(layer_one[0].0, 0.1 - (2.0f32).powi(-4));

            let layer_zero = heap.layer_estimated_distances(0);
            println!("Layer 0: {:?}", layer_zero);
            assert!(layer_zero.len() == 3);
            assert_approx_eq!(layer_zero[0].0, 0.0);
            assert_approx_eq!(layer_zero[1].0, 0.0);
            assert!(layer_zero[2].1 == 8);
            assert_approx_eq!(layer_zero[2].0, 0.1);

            let unpack = heap.unpack();
            println!("unpack: {:?}", unpack);

            for i in 1..5 {
                assert!(unpack[i - 1].1 == i as u64);
            }
        }
    */
}
