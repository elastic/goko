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

use crate::NodeAddress;
use std::collections::{BinaryHeap, HashMap};
use std::f32;

use super::query_items::{QueryAddress};

/// This is used to find the closest `k` nodes to the query point, either to get summary statistics out, or
/// to restrict a Gaussian Mixture Model
#[derive(Debug)]
pub struct TraceQueryHeap {
    layer_max_heaps: HashMap<i32,BinaryHeap<QueryAddressRev>>,
    layer_min_heaps: HashMap<i32,BinaryHeap<QueryAddress>>,
    k: usize,
    scale_base: f32,
}

impl TraceQueryHeap {
    /// Shoves data in here.
    pub fn push_nodes(
        &mut self,
        indexes: &[NodeAddress],
        dists: &[f32],
    ) {
        for ((si, pi), d) in indexes.iter().zip(dists) {
            let emd = (d - self.scale_base.powi(*si)).max(0.0);

            let mut heap = self.layer_heaps.entry(*si).or_insert(BinaryHeap::new());
            heap.push(QueryAddressRev {
                address: (*si, *pi),
                dist_to_center: *d,
                min_dist: emd,
            });
            while heap.len() > self.k {
                heap.pop();
            }
        }
    }

    /// The count at a layer
    pub fn count(&self, si:i32) -> usize {
        match self.layer_heaps.get(si) {
            Some(heap) => heap.len(),
            None => 0,
        }
    } 
}