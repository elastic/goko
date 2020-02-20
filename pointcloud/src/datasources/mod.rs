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

//! Some data sources and a trait to dimension and uniformly reference the data contained. 
//! The only currently supported are memmaps and ram blobs.

use std::fmt::Debug;
use crate::errors::*;

#[allow(dead_code)]
mod memmapf32;
mod memmap;

#[doc(hidden)]
pub use memmap::*;

/// The trait for uniform access across many data types
pub trait DataSource: Send + Sync + Debug {
    /// Make this panic with `DataAccessError` when implementing, 
    fn get(&self, i: usize) -> Result<&[f32], PointCloudError>;
    /// The dimension that this dimensioned data respects
    fn dim(&self) -> usize;
    /// This should always be the number of points contained in this portion of the dataset.
    fn len(&self) -> usize;
    /// The name of this portion of the dataset. Useful for reporting errors to the user. 
    fn name(&self) -> String;
}