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

#[cfg(test)]
#[macro_use]
extern crate assert_approx_eq;


mod distances;
pub use distances::*;
pub mod errors;

pub mod data_sources;
pub mod label_sources;
pub mod summaries;

pub mod glued_data_cloud;
pub mod loaders;

mod base_traits;
#[doc(inline)]
pub use base_traits::*;

use label_sources::SmallIntLabels;
use data_sources::DataRam;

/// A sensible default for an labeled cloud
pub type DefaultLabeledCloud<M> = SimpleLabeledCloud<DataRam<M>,SmallIntLabels>;
/// A sensible default for an unlabeled cloud
pub type DefaultCloud<M> = DataRam<M>;

/// To make things more obvious, we type the point index.
/// This is abstracted over the files that were used to build the point cloud
pub type PointIndex = usize;
/// To make things more obvious, we type the point name that we pull from the label CSV
pub type PointName = String;

/// Reference to a point inside of a dataset.
#[derive(Clone, Copy, Debug)]
pub enum PointRef<'a> {
    /// Dense reference
    Dense(&'a [f32]),
    /// Sparse reference, values, then indexes
    Sparse(&'a [f32], &'a [u32]),
}

/// An actual point, self contained with it's own objects on the heap.
#[derive(Clone, Debug)]
pub enum Point {
    /// Dense contiguous point
    Dense(Vec<f32>),
    /// Sparse contiguous point, values then indexes
    Sparse(Vec<f32>, Vec<u32>),
}

impl Point {
    /// Borrows a point. Should be an implementation of `AsRef`, but the lifetimes disagreed.
    pub fn to_ref<'a, 'b: 'a>(&'b self) -> PointRef<'a> {
        match self {
            Point::Dense(v) => PointRef::Dense(&v[..]),
            Point::Sparse(v, i) => PointRef::Sparse(&v[..], &i[..]),
        }
    }
}
