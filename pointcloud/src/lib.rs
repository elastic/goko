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
pub mod pc_errors;

pub mod data_sources;
pub mod label_sources;
pub mod summaries;

pub mod glued_data_cloud;
pub mod loaders;

mod base_traits;
#[doc(inline)]
pub use base_traits::*;

use data_sources::DataRam;
use label_sources::SmallIntLabels;

/// A sensible default for an labeled cloud
pub type DefaultLabeledCloud<M> = SimpleLabeledCloud<DataRam<M>, SmallIntLabels>;
/// A sensible default for an unlabeled cloud
pub type DefaultCloud<M> = DataRam<M>;

impl<M: Metric> DefaultLabeledCloud<M> {
    /// Simple way of gluing together the most common data source
    pub fn new_simple(data: Vec<f32>, dim: usize, labels: Vec<u64>) -> DefaultLabeledCloud<M> {
        SimpleLabeledCloud::new(
            DataRam::<M>::new(data, dim).unwrap(),
            SmallIntLabels::new(labels, None),
        )
    }
}

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

///
pub struct DenseIter<'a> {
    p_ref: PointRef<'a>,
    index: usize,
    sparse_index: usize,
    dim: usize,
}

impl<'a> Iterator for DenseIter<'a> {
    type Item = f32;
    fn next(&mut self) -> Option<Self::Item> {
        match self.p_ref {
            PointRef::Dense(vals) => {
                if self.index < vals.len() {
                    self.index += 1;
                    Some(vals[self.index - 1])
                } else {
                    None
                }
            }
            PointRef::Sparse(vals, inds) => {
                if self.index < self.dim {
                    if inds[self.sparse_index] == self.index as u32 {
                        self.sparse_index += 1;
                        self.index += 1;
                        Some(vals[self.sparse_index - 1])
                    } else if self.index < self.dim {
                        self.index += 1;
                        Some(0.0)
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        match self.p_ref {
            PointRef::Dense(vals) => (vals.len(), Some(vals.len())),
            PointRef::Sparse(_, _) => (self.dim, Some(self.dim)),
        }
    }
}

impl<'a> PointRef<'a> {
    /// Gives an iterator that lets you treat the point reference as a dense vector
    pub fn dense_iter(&self, dim: usize) -> DenseIter<'a> {
        DenseIter {
            p_ref: self.into(),
            index: 0,
            sparse_index: 0,
            dim,
        }
    }
}

/// An actual point, self contained with it's own objects on the heap.
#[derive(Clone, Debug)]
pub enum Point {
    /// Dense contiguous point
    Dense(Vec<f32>),
    /// Sparse contiguous point, values then indexes
    Sparse(Vec<f32>, Vec<u32>),
}

impl<'a> From<&'a [f32]> for PointRef<'a> {
    fn from(arr: &'a [f32]) -> PointRef<'a> {
        PointRef::Dense(arr)
    }
}

impl<'a, T: AsRef<[f32]>> From<&'a T> for PointRef<'a> {
    fn from(arr: &'a T) -> PointRef<'a> {
        PointRef::Dense(arr.as_ref())
    }
}

impl<'b, 'a: 'b> From<&'b PointRef<'a>> for PointRef<'a> {
    fn from(arr: &'b PointRef<'a>) -> PointRef<'a> {
        match arr {
            PointRef::Dense(v) => PointRef::Dense(&v[..]),
            PointRef::Sparse(v, i) => PointRef::Sparse(&v[..], &i[..]),
        }
    }
}

impl<'b, 'a: 'b, 'c: 'b> From<&'c &'b PointRef<'a>> for PointRef<'a> {
    fn from(arr: &'c &'b PointRef<'a>) -> PointRef<'a> {
        match arr {
            PointRef::Dense(v) => PointRef::Dense(&v[..]),
            PointRef::Sparse(v, i) => PointRef::Sparse(&v[..], &i[..]),
        }
    }
}

impl<'a> From<(&'a [f32], &'a [u32])> for PointRef<'a> {
    fn from(arr: (&'a [f32], &'a [u32])) -> PointRef<'a> {
        PointRef::Sparse(arr.0, arr.1)
    }
}

impl<'a> From<&'a Point> for PointRef<'a> {
    fn from(arr: &'a Point) -> PointRef<'a> {
        match arr {
            Point::Dense(v) => PointRef::Dense(&v[..]),
            Point::Sparse(v, i) => PointRef::Sparse(&v[..], &i[..]),
        }
    }
}

