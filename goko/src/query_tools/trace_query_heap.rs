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

use super::*;
use crate::NodeAddress;
use std::collections::{BinaryHeap, HashMap};
use std::f32;

use super::query_items::{QueryAddress, QueryAddressRev};

/// This is used to find the closest `k` nodes to the query point, either to get summary statistics out, or
/// to restrict a Gaussian Mixture Model
#[derive(Debug)]
pub struct MultiscaleQueryHeap {
    layer_max_heaps: HashMap<i32, BinaryHeap<QueryAddressRev>>,
    layer_min_heaps: HashMap<i32, BinaryHeap<QueryAddress>>,
    k: usize,
    scale_base: f32,
}

impl RoutingQueryHeap for MultiscaleQueryHeap {
    /// Shoves data in here.
    fn push_nodes(
        &mut self,
        indexes: &[NodeAddress],
        dists: &[f32],
        _parent_address: Option<NodeAddress>,
    ) {
        for ((si, pi), d) in indexes.iter().zip(dists) {
            let emd = (d - self.scale_base.powi(*si)).max(0.0);

            println!("\t Inserting {:?} into max heap", ((si, pi), d));
            let max_heap = self
                .layer_max_heaps
                .entry(*si)
                .or_insert_with(BinaryHeap::new);
            max_heap.push(QueryAddressRev {
                address: (*si, *pi),
                dist_to_center: *d,
                min_dist: emd,
            });
            while max_heap.len() > self.k + 1 {
                max_heap.pop();
            }

            let min_heap = self
                .layer_min_heaps
                .entry(*si)
                .or_insert_with(BinaryHeap::new);
            min_heap.push(QueryAddress {
                address: (*si, *pi),
                dist_to_center: *d,
                min_dist: emd,
            });
        }
    }
}

impl MultiscaleQueryHeap {
    /// Creates a new set of heaps, hashmaps, and parameters designed to do multiscale KNN
    pub fn new(k: usize, scale_base: f32) -> MultiscaleQueryHeap {
        MultiscaleQueryHeap {
            layer_max_heaps: HashMap::new(),
            layer_min_heaps: HashMap::new(),
            k,
            scale_base,
        }
    }

    /// Gives us the closest unqueried node on a particular layer
    pub fn pop_closest_unqueried(&mut self, scale_index: i32) -> Option<(f32, NodeAddress)> {
        match self.layer_min_heaps.get_mut(&scale_index) {
            Some(heap) => match heap.pop() {
                None => None,
                Some(qa) => {
                    let max_dist = self
                        .furthest_node(scale_index)
                        .map(|(d, _)| d)
                        .unwrap_or(0.0);
                    if max_dist <= qa.min_dist {
                        Some((qa.min_dist, qa.address))
                    } else {
                        None
                    }
                }
            },
            None => None,
        }
    }

    /// Unpacks this to a digestible format
    pub fn unpack(mut self) -> HashMap<i32, Vec<(f32, NodeAddress)>> {
        self.layer_max_heaps
            .drain()
            .map(|(si, heap)| {
                let mut v: Vec<(f32, NodeAddress)> = heap
                    .into_iter_sorted()
                    .map(|qa| (qa.dist_to_center, qa.address))
                    .collect();
                v.reverse();
                (si, v)
            })
            .collect()
    }

    /// returns the node on a layer that is the furthest away. This returns None if the heap isn't full (less than K elements)
    pub fn furthest_node(&self, scale_index: i32) -> Option<(f32, NodeAddress)> {
        self.layer_max_heaps
            .get(&scale_index)
            .map(|max_heap| {
                if max_heap.len() < self.k {
                    None
                } else {
                    max_heap.peek().map(|x| (x.dist_to_center, x.address))
                }
            })
            .flatten()
    }

    /// The count at a layer
    pub fn count(&self, si: i32) -> usize {
        match self.layer_max_heaps.get(&si) {
            Some(heap) => heap.len(),
            None => 0,
        }
    }
}

//Tested in the node file too
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    #[test]
    fn multiscale_insertion_unpacks_correctly() {
        let mut trace_heap = MultiscaleQueryHeap::new(5, 2.0);
        let dists = [0.1, 0.2, 0.4, 0.5, 0.1, 0.2, 0.4, 0.5, 0.05];
        let addresses = [
            (0, 0),
            (0, 1),
            (0, 3),
            (0, 4),
            (1, 0),
            (1, 1),
            (1, 3),
            (1, 4),
            (1, 2),
        ];
        trace_heap.push_nodes(&addresses, &dists, None);
        println!("{:#?}", trace_heap);
        let results = trace_heap.unpack();
        let layer_0 = results.get(&0).unwrap();
        assert_eq!(layer_0[0], (0.1, (0, 0)));
        assert_eq!(layer_0[1], (0.2, (0, 1)));

        let layer_1 = results.get(&1).unwrap();
        assert_eq!(layer_1[0], (0.05, (1, 2)));
        assert_eq!(layer_1[1], (0.1, (1, 0)));
        assert_eq!(layer_1[2], (0.2, (1, 1)));
    }
}
