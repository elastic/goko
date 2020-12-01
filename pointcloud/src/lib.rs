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
//! # Point Cloud
//! Abstracts data access over several files and glues metadata files to vector data files

#![allow(dead_code)]
//#![deny(warnings)]
#![warn(missing_docs)]
#![allow(clippy::cast_ptr_alignment)]
#![feature(result_flattening)]
#![feature(is_sorted)]
#![feature(iterator_fold_self)]
#![feature(generic_associated_types)]

#[cfg(test)]
#[macro_use]
extern crate assert_approx_eq;

pub mod pc_errors;

mod base_traits;
#[doc(inline)]
pub use base_traits::*;
pub mod metrics;

pub mod points;

pub mod data_sources;

//pub mod glued_data_cloud;

pub mod label_sources;
pub mod summaries;

//pub mod loaders;
/*
use data_sources::DataRamL2;
use label_sources::SmallIntLabels;

/// A sensible default for an labeled cloud
pub type DefaultLabeledCloud = SimpleLabeledCloud<DataRamL2, SmallIntLabels>;
/// A sensible default for an unlabeled cloud
pub type DefaultCloud = DataRamL2;

impl DefaultLabeledCloud {
    /// Simple way of gluing together the most common data source
    pub fn new_simple(data: Vec<f32>, dim: usize, labels: Vec<i64>) -> DefaultLabeledCloud {
        SimpleLabeledCloud::new(
            DataRamL2::new(data, dim).unwrap(),
            SmallIntLabels::new(labels, None),
        )
    }
}
*/