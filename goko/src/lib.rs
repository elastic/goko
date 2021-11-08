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
#![doc(test(attr(allow(unused_variables), deny(warnings))))]
#![feature(binary_heap_into_iter_sorted)]
#![feature(associated_type_defaults)]

//! # Goko
//! This is an lock-free efficient implementation of a covertree for data science. The traditional
//! application of this is for KNN, but it can be applied and used in lots of other applications.
//!
//! ## Parameter Guide
//! The structure is controlled by 3 parameters, the most important of which
//! is the scale_base. This should be between 1.2 and 2ish. A higher value will create more outliers.
//! Outliers are not loaded into ram at startup, but a high value slows down creation of a tree
//! significantly. Theoretically, this value doesn't matter to the big O time, but I wouldn't go above 2.
//!
//! The cutoff value controls how many points a leaf is allowed to cover. A smaller value gives faster
//! bottom level queries, but at the cost of higher memory useage. Do not expect a value of 100 will give
//! 1/100 the memory useage of a value of 1. It'd be closer to 1/10 or 1/5th. This is because the distribution
//! of the number of children of a node. A high cutoff will increase the compute of the true-knn by a little bit.
//!
//! The resolution is the minimum scale index, this again reduces memory footprint and increases the query time
//! for true KNN.
//! Once a node's resolution dips below this value we stop and covert the remaining coverage into a leaf.
//! This is mainly to stop before floating point errors become an issue. Try to choose it to result in a cutoff of about
//! 2^-9.
//!
//! See the git readme for a description of the algo.
//!

#[cfg(test)]
#[macro_use]
extern crate smallvec;

extern crate rand;

use rayon::prelude::*;
#[cfg(test)]
#[macro_use]
extern crate assert_approx_eq;

use pointcloud::*;
pub mod errors;
pub use errors::GokoResult;

pub(crate) mod monomap;

mod covertree;
pub use covertree::*;

pub mod query_interface;

mod tree_file_format;
pub mod utils;

pub mod plugins;

/// The data structure explicitly seperates the covertree by layer, and the addressing schema for nodes
/// is a pair for the layer index and the center point index of that node.
pub use core_goko::*;
