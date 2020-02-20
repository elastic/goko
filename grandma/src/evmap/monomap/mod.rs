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

//! A lock free, eventually consistent, concurrent single-value map.
//! This is almost identical to the multimap, but duplicates the data.
//! This stores the values in 2 copies of the hashmap, so it's meant to be used when you care about speed and 
//! concurrency with updates rather than memory efficency. Ideally if your type has any large vectors you'd partner 
//! this with a multimap to store the vector, and keep the complex logic on this structure.


use std::collections::hash_map::RandomState;
use std::fmt;
use std::hash::{BuildHasher, Hash};
use std::sync::Arc;

use crate::evmap::inner::Inner;

/// Merge object for reducing the entries of a entry to a single one.
pub struct Updater<V>(pub(crate) Box<dyn Fn(&mut V) + Send + Sync>);

impl<V> Updater<V> {
    /// Evaluate the predicate for the given element
    #[inline]
    pub fn eval(&self, value: &mut V) {
        (*self.0)(value)
    }
}

impl<V> PartialEq for Updater<V> {
    #[inline]
    fn eq(&self, other: &Self) -> bool {
        &*self.0 as *const _ == &*other.0 as *const _
    }
}

impl<V> Eq for Updater<V> {}

impl<V> fmt::Debug for Updater<V> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_tuple("Updater")
            .field(&format_args!("{:p}", &*self.0 as *const _))
            .finish()
    }
}

/// A pending map operation.
///
/// Note that this enum should be considered
/// [non-exhaustive](https://github.com/rust-lang/rust/issues/44109).
#[derive(PartialEq, Eq, Debug)]
// TODO: #[non_exhaustive]
// https://github.com/rust-lang/rust/issues/44109
pub enum MonoOperation<K, V> {
    /// Add this value to the set of entries for this key.
    Insert(K, V),
    /// Add this value to the set of entries for this key.
    Update(K, Updater<V>),
    /// Remove the value for this key.
    Remove(K),
    /// Remove all values for all keys.
    ///
    /// Note that this will iterate once over all the keys internally.
    Purge,
    // Since we have a feature that adds an enum variant, features are only additive (as they need
    // to be) if users never try to exhaustively match on this enum. Once rust-lang/rust#44109
    // lands, we'll have a more standard way to do this, but for now we rely on this trick:
    #[doc(hidden)]
    __Nonexhaustive,
}

mod write;
pub use crate::evmap::monomap::write::MonoWriteHandle;

mod read;
pub use crate::evmap::monomap::read::{MonoReadHandle, MonoReadHandleFactory};

/// Options for how to initialize the map.
///
/// In particular, the options dictate the hashing function, meta type, and initial capacity of the
/// map.
pub struct MonoOptions<M, S>
where
    S: BuildHasher,
{
    meta: M,
    hasher: S,
    capacity: Option<usize>,
}

impl Default for MonoOptions<(), RandomState> {
    fn default() -> Self {
        MonoOptions {
            meta: (),
            hasher: RandomState::default(),
            capacity: None,
        }
    }
}

impl<M, S> MonoOptions<M, S>
where
    S: BuildHasher,
{
    /// Set the initial meta value for the map.
    pub fn with_meta<M2>(self, meta: M2) -> MonoOptions<M2, S> {
        MonoOptions {
            meta,
            hasher: self.hasher,
            capacity: self.capacity,
        }
    }

    /// Set the hasher used for the map.
    pub fn with_hasher<S2>(self, hash_builder: S2) -> MonoOptions<M, S2>
    where
        S2: BuildHasher,
    {
        MonoOptions {
            meta: self.meta,
            hasher: hash_builder,
            capacity: self.capacity,
        }
    }

    /// Set the initial capacity for the map.
    pub fn with_capacity(self, capacity: usize) -> MonoOptions<M, S> {
        MonoOptions {
            meta: self.meta,
            hasher: self.hasher,
            capacity: Some(capacity),
        }
    }

    /// Create the map, and construct the read and write handles used to access it.
    #[allow(clippy::type_complexity)]
    pub fn construct<K, V>(self) -> (MonoReadHandle<K, V, M, S>, MonoWriteHandle<K, V, M, S>)
    where
        K: Eq + Hash + Clone,
        S: BuildHasher + Clone,
        V: Clone,
        M: 'static + Clone,
    {
        let epochs = Default::default();
        let inner = if let Some(cap) = self.capacity {
            Inner::with_capacity_and_hasher(self.meta, cap, self.hasher)
        } else {
            Inner::with_hasher(self.meta, self.hasher)
        };

        let mut w_handle = inner.clone();
        w_handle.mark_ready();
        let r = read::new(inner, Arc::clone(&epochs));
        let w = write::new(w_handle, epochs, r.clone());
        (r, w)
    }
}

/// Create an empty eventually consistent map.
///
/// Use the [`Options`](./struct.Options.html) builder for more control over initialization.
#[allow(clippy::type_complexity)]
pub fn new<K, V>() -> (
    MonoReadHandle<K, V, (), RandomState>,
    MonoWriteHandle<K, V, (), RandomState>,
)
where
    K: Eq + Hash + Clone,
    V: Clone,
{
    MonoOptions::default().construct()
}

/// Create an empty eventually consistent map with meta information.
///
/// Use the [`Options`](./struct.Options.html) builder for more control over initialization.
#[allow(clippy::type_complexity)]
pub fn with_meta<K, V, M>(
    meta: M,
) -> (
    MonoReadHandle<K, V, M, RandomState>,
    MonoWriteHandle<K, V, M, RandomState>,
)
where
    K: Eq + Hash + Clone,
    V: Clone,
    M: 'static + Clone,
{
    MonoOptions::default().with_meta(meta).construct()
}

/// Create an empty eventually consistent map with meta information and custom hasher.
///
/// Use the [`Options`](./struct.Options.html) builder for more control over initialization.
#[allow(clippy::type_complexity)]
pub fn with_hasher<K, V, M, S>(
    meta: M,
    hasher: S,
) -> (MonoReadHandle<K, V, M, S>, MonoWriteHandle<K, V, M, S>)
where
    K: Eq + Hash + Clone,
    V: Clone,
    M: 'static + Clone,
    S: BuildHasher + Clone,
{
    MonoOptions::default()
        .with_hasher(hasher)
        .with_meta(meta)
        .construct()
}

// test that MonoReadHandle isn't Sync
// waiting on https://github.com/rust-lang/rust/issues/17606
//#[test]
//fn is_not_sync() {
//    use std::sync;
//    use std::thread;
//    let (r, mut w) = new();
//    w.insert(true, false);
//    let x = sync::Arc::new(r);
//    thread::spawn(move || { drop(x); });
//}
