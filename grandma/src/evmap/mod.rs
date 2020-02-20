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

#![deny(missing_docs)]

//! A memory efficient multi-value map in `multimap` with a memory inefficient single value map in `monomap`

use std::sync::{atomic, Arc, Mutex};

mod inner;
pub mod shallow_copy;
pub use crate::evmap::shallow_copy::ShallowCopy;

pub(crate) type Epochs = Arc<Mutex<Vec<Arc<atomic::AtomicUsize>>>>;

pub mod monomap;
