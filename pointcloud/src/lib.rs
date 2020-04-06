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



#[macro_use]
extern crate serde;

mod point_cloud;
#[doc(inline)]
pub use point_cloud::PointCloud;

mod distances;
pub use distances::*;
pub mod errors;

pub mod labels;
pub mod utils;

pub mod datasources;
pub use datasources::DataSource;

/// To make things more obvious, we type the point index.
/// This is abstracted over the files that were used to build the point cloud
pub type PointIndex = u64;
/// To make things more obvious, we type the point name that we pull from the label CSV
pub type PointName = String;
