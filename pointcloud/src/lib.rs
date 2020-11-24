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
/*
mod distances;
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
    pub fn new_simple(data: Vec<f32>, dim: usize, labels: Vec<i64>) -> DefaultLabeledCloud<M> {
        SimpleLabeledCloud::new(
            DataRam::<M>::new(data, dim).unwrap(),
            SmallIntLabels::new(labels, None),
        )
    }
}
*/
pub use distances::*;

/// To make things more obvious, we type the point index.
/// This is abstracted over the files that were used to build the point cloud
pub type PointIndex = usize;

pub trait Point<T> {
    type DenseIter: Iterator<Item = T>;
    fn dist(&self,other: &Self) -> T;
    fn norm(&self) -> T;
    fn dense(&self) -> Vec<T>;
    fn dense_iter(&self) -> Self::DenseIter;
}

struct DenseL2<'a, T> {
    data: &'a [T],
}

impl<'a> Point<f32> for DenseL2<'a, f32> {
    type DenseIter =  std::iter::Cloned<std::slice::Iter<'a,f32>>;

    fn dist(&self,other: &Self) -> f32 {
        self.data.iter().zip(other.data).map(|(x,y)| x - y).map(|z| z*z).fold(0.0,|a,x| a+x).sqrt()
    }
    fn norm(&self) -> f32 {
        self.data.iter().map(|z| z*z).fold(0.0,|a,x| a+x).sqrt()
    }
    fn dense(&self) -> Vec<f32> {
        Vec::from(self.data)
    }
    fn dense_iter(&self) -> Self::DenseIter {
        self.data.iter().cloned()
    }
}

impl<'a> Point<f64> for DenseL2<'a, f64> {
    type DenseIter =  std::iter::Cloned<std::slice::Iter<'a,f64>>;

    fn dist(&self,other: &Self) -> f64 {
        self.data.iter().zip(other.data).map(|(x,y)| x - y).map(|z| z*z).fold(0.0,|a,x| a+x).sqrt()
    }
    fn norm(&self) -> f64 {
        self.data.iter().map(|z| z*z).fold(0.0,|a,x| a+x).sqrt()
    }
    fn dense(&self) -> Vec<f64> {
        self.data.iter().map(|x| *x).collect()
    }
    fn dense_iter(&self) -> Self::DenseIter {
        self.data.iter().cloned()
    }
}

struct SparseL2<'a, T, S> {
    dim: S,
    values: &'a [T],
    indexes: &'a [S],
}

impl<'a> Point<f32> for SparseL2<'a, f32, u32> {
    fn dist(&self,other: &Self) -> f32 {
        L2::sparse(self.indexes, self.values, other.indexes, other.values)
    }
    fn norm(&self) -> f32 {
        L2::norm(self.values)
    }
    fn dense(&self) -> Vec<f32> {
        let mut v = vec![0.0;self.dim as usize];
        for (v,i) in values.iter().zip(self.items) 
    }
}