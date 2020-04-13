use std::hash::{BuildHasher, Hash};

use hashbrown::HashMap as MapImpl;

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
