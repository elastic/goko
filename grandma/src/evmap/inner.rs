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

use std::hash::{BuildHasher, Hash};

#[cfg(feature = "indexed")]
use indexmap::IndexMap as MapImpl;
#[cfg(not(feature = "indexed"))]
use std::collections::HashMap as MapImpl;

#[cfg(not(feature = "smallvec"))]
pub(crate) type Values<T> = Vec<T>;

#[cfg(feature = "smallvec")]
pub(crate) type Values<T> = smallvec::SmallVec<[T; 1]>;

pub(crate) struct Inner<K, T, M, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    pub(crate) data: MapImpl<K, T, S>,
    pub(crate) meta: M,
    ready: bool,
}

impl<K, T, M, S> Clone for Inner<K, T, M, S>
where
    K: Eq + Hash + Clone,
    S: BuildHasher + Clone,
    M: Clone,
{
    fn clone(&self) -> Self {
        assert!(self.data.is_empty());
        Inner {
            data: MapImpl::with_capacity_and_hasher(
                self.data.capacity(),
                self.data.hasher().clone(),
            ),
            meta: self.meta.clone(),
            ready: self.ready,
        }
    }
}

impl<K, T, M, S> Inner<K, T, M, S>
where
    K: Eq + Hash,
    S: BuildHasher,
{
    pub fn with_hasher(m: M, hash_builder: S) -> Self {
        Inner {
            data: MapImpl::with_hasher(hash_builder),
            meta: m,
            ready: false,
        }
    }

    pub fn with_capacity_and_hasher(m: M, capacity: usize, hash_builder: S) -> Self {
        Inner {
            data: MapImpl::with_capacity_and_hasher(capacity, hash_builder),
            meta: m,
            ready: false,
        }
    }

    pub fn mark_ready(&mut self) {
        self.ready = true;
    }

    pub fn is_ready(&self) -> bool {
        self.ready
    }
}
