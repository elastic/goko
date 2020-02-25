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

#![allow(dead_code)]
#![deny(warnings)]
#![warn(missing_docs)]
#![doc(test(attr(allow(unused_variables), deny(warnings))))]

//! # Cuddly Raccoon
//! This is an efficient implementation of a covertree
//! It's structure is controlled by 3 parameters, the most important of which
//! is the scale_base. This should be between 1.2 and 2ish. A higher value will create more outliers.
//! Outliers are not loaded into ram at startup, but a high value slows down creation of a tree
//! significantly. Theoretically, this value doesn't matter to the big O time, but I wouldn't go above 2.
//!
//! The cutoff value controls how many points a leaf is allowed to cover. A smaller value gives faster
//! bottom level queries, but at the cost of higher memory useage. Do not expect a value of 100 will give
//! 1/100 the memory useage of a value of 1. It'd be closer to 1/10 or 1/5th. This is because the number of c.
//! This will increase the compute by a little bit (maybe 2x for a knn with a very high value).
//!
//! The resolution is the minimum scale index, this again reduces memory footprint and increases the query time.
//! Once a node's resolution dips below this value we stop and covert the remaining coverage into a leaf.
//!
//! See the git readme for a description of the algo.
//!

#[cfg(test)]
#[macro_use]
extern crate smallvec;

use rayon::prelude::*;
#[cfg(test)]
#[macro_use]
extern crate assert_approx_eq;

use pointcloud::*;
pub mod errors;
pub use errors::GrandmaResult;

pub(crate) mod evmap;

mod tree_file_format;
mod builders;
mod data_caches;
pub mod layer;
pub mod node;
pub mod query_tools;
mod tree;
pub mod utils;

pub use builders::CoverTreeBuilder;
pub use tree::*;

/// The data structure explicitly seperates the covertree by layer, and the addressing schema for nodes 
/// is a pair for the layer index and the center point index of that node. 
pub type NodeAddress = (i32, PointIndex);
/// Like with a node address, the clusters are segmented by layer so we also reference by layer. The ClusterID is not meaningful, it's just a uint.
pub type ClusterAddress = (i32, usize);
