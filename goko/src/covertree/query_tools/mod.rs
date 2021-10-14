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

pub(crate) mod query_items;

pub(crate) mod knn_query_heap;
pub use knn_query_heap::KnnQueryHeap;

/// If you have a algorithm that does local brute force KNN on just the children,
/// implement this to use the node fn
pub trait RoutingQueryHeap {
    /// Shoves data in.
    fn push_nodes(
        &mut self,
        indexes: &[NodeAddress],
        dists: &[f32],
        parent_address: Option<NodeAddress>,
    );
}

/// If you have a algorithm that does local brute force KNN on just the singletons,
/// implement this to use the node fn
pub trait SingletonQueryHeap {
    /// Shove a bunch of single points onto the heap
    fn push_outliers(&mut self, indexes: &[usize], dists: &[f32]);
}
