use super::{MonoOperation, Updater};
use crate::evmap::inner::Inner;
use crate::evmap::monomap::read::MonoReadHandle;

use std::collections::hash_map::RandomState;
use std::hash::{BuildHasher, Hash};
use std::sync::atomic;
use std::sync::{Arc, MutexGuard};
use std::{mem, thread};

pub struct MonoWriteHandle<K, V, M = (), S = RandomState>
where
    K: Eq + Hash + Clone,
    S: BuildHasher + Clone,
    V: Clone,
    M: 'static + Clone,
{
    epochs: crate::evmap::Epochs,
    w_handle: Option<Box<Inner<K, V, M, S>>>,
    oplog: Vec<MonoOperation<K, V>>,
    swap_index: usize,
    r_handle: MonoReadHandle<K, V, M, S>,
    last_epochs: Vec<usize>,
    meta: M,
    first: bool,
    second: bool,
}

pub(crate) fn new<K, V, M, S>(
    w_handle: Inner<K, V, M, S>,
    epochs: crate::evmap::Epochs,
    r_handle: MonoReadHandle<K, V, M, S>,
) -> MonoWriteHandle<K, V, M, S>
where
    K: Eq + Hash + Clone,
    V: Clone,
    S: BuildHasher + Clone,
    M: 'static + Clone,
{
    let m = w_handle.meta.clone();
    MonoWriteHandle {
        epochs,
        w_handle: Some(Box::new(w_handle)),
        oplog: Vec::new(),
        swap_index: 0,
        r_handle,
        last_epochs: Vec::new(),
        meta: m,
        first: true,
        second: false,
    }
}

impl<K, V, M, S> Drop for MonoWriteHandle<K, V, M, S>
where
    K: Eq + Hash + Clone,
    S: BuildHasher + Clone,
    V: Clone,
    M: 'static + Clone,
{
    fn drop(&mut self) {
        use std::ptr;

        // first, ensure both maps are up to date
        // (otherwise safely dropping deduplicated rows is a pain)
        if !self.oplog.is_empty() {
            self.refresh();
        }
        if !self.oplog.is_empty() {
            self.refresh();
        }
        assert!(self.oplog.is_empty());

        // next, grab the read handle and set it to NULL
        let r_handle = self
            .r_handle
            .inner
            .swap(ptr::null_mut(), atomic::Ordering::Release);

        // now, wait for all readers to depart
        let epochs = Arc::clone(&self.epochs);
        let mut epochs = epochs.lock().unwrap();
        self.wait(&mut epochs);

        // ensure that the subsequent epoch reads aren't re-ordered to before the swap
        atomic::fence(atomic::Ordering::SeqCst);

        let w_handle = &mut self.w_handle.as_mut().unwrap().data;

        #[cfg(not(feature = "indexed"))]
        w_handle.drain();
        #[cfg(feature = "indexed")]
        w_handle.drain(..);

        // then we drop r_handle, which will free all the records. this is safe, since we know that
        // no readers are using this pointer anymore (due to the .wait() following swapping the
        // pointer with NULL).
        drop(unsafe { Box::from_raw(r_handle) });
    }
}

impl<K, V, M, S> MonoWriteHandle<K, V, M, S>
where
    K: Eq + Hash + Clone,
    S: BuildHasher + Clone,
    V: Clone,
    M: 'static + Clone,
{
    fn wait(&mut self, epochs: &mut MutexGuard<Vec<Arc<atomic::AtomicUsize>>>) {
        let mut iter = 0;
        let mut starti = 0;
        let high_bit = 1usize << (mem::size_of::<usize>() * 8 - 1);
        self.last_epochs.resize(epochs.len(), 0);
        'retry: loop {
            // read all and see if all have changed (which is likely)
            for (i, epoch) in epochs.iter().enumerate().skip(starti) {
                if self.last_epochs[i] & high_bit != 0 {
                    // reader was not active right after last swap
                    // and therefore *must* only see new pointer
                    continue;
                }

                let now = epoch.load(atomic::Ordering::Acquire);
                if (now != self.last_epochs[i]) | (now & high_bit != 0) | (now == 0) {
                    // reader must have seen last swap
                } else {
                    // reader may not have seen swap
                    // continue from this reader's epoch
                    starti = i;

                    // how eagerly should we retry?
                    if iter != 20 {
                        iter += 1;
                    } else {
                        thread::yield_now();
                    }

                    continue 'retry;
                }
            }
            break;
        }
    }

    /// Refresh the handle used by readers so that pending writes are made visible.
    ///
    /// This method needs to wait for all readers to move to the new handle so that it can replay
    /// the operational log onto the stale map copy the readers used to use. This can take some
    /// time, especially if readers are executing slow operations, or if there are many of them.
    pub fn refresh(&mut self) -> &mut Self {
        // we need to wait until all epochs have changed since the swaps *or* until a "finished"
        // flag has been observed to be on for two subsequent iterations (there still may be some
        // readers present since we did the previous refresh)
        //
        // NOTE: it is safe for us to hold the lock for the entire duration of the swap. we will
        // only block on pre-existing readers, and they are never waiting to push onto epochs
        // unless they have finished reading.
        let epochs = Arc::clone(&self.epochs);
        let mut epochs = epochs.lock().unwrap();

        self.wait(&mut epochs);

        {
            // all the readers have left!
            // we can safely bring the w_handle up to date.
            let w_handle = self.w_handle.as_mut().unwrap();

            if self.second {
                // before the first refresh, all writes went directly to w_handle. then, at the
                // first refresh, r_handle and w_handle were swapped. thus, the w_handle we
                // have now is empty, *and* none of the writes in r_handle are in the oplog.
                // we therefore have to first clone the entire state of the current r_handle
                // and make that w_handle, and *then* replay the oplog (which holds writes
                // following the first refresh).
                //
                // this may seem unnecessarily complex, but it has the major advantage that it
                // is relatively efficient to do lots of writes to the evmap at startup to
                // populate it, and then refresh().
                let r_handle = unsafe {
                    self.r_handle
                        .inner
                        .load(atomic::Ordering::Relaxed)
                        .as_mut()
                        .unwrap()
                };

                // XXX: it really is too bad that we can't just .clone() the data here and save
                // ourselves a lot of re-hashing, re-bucketization, etc.
                w_handle.data = r_handle.data.clone();
            }

            // the w_handle map has not seen any of the writes in the oplog
            // the r_handle map has not seen any of the writes following swap_index
            if self.swap_index != 0 {
                // we can drain out the operations that only the w_handle map needs
                //
                // NOTE: the if above is because drain(0..0) would remove 0
                //
                // NOTE: the distinction between apply_first and apply_second is the reason why our
                // use of shallow_copy is safe. we apply each op in the oplog twice, first with
                // apply_first, and then with apply_second. on apply_first, no destructors are
                // called for removed values (since those values all still exist in the other map),
                // and all new values are shallow copied in (since we need the original for the
                // other map). on apply_second, we call the destructor for anything that's been
                // removed (since those removals have already happened on the other map, and
                // without calling their destructor).
                for op in self.oplog.drain(0..self.swap_index) {
                    Self::apply_second(w_handle, op);
                }
            }
            // the rest have to be cloned because they'll also be needed by the r_handle map
            for op in self.oplog.iter_mut() {
                Self::apply_first(w_handle, op);
            }
            // the w_handle map is about to become the r_handle, and can ignore the oplog
            self.swap_index = self.oplog.len();
            // ensure meta-information is up to date
            w_handle.meta = self.meta.clone();
            w_handle.mark_ready();

            // w_handle (the old r_handle) is now fully up to date!
        }

        // at this point, we have exclusive access to w_handle, and it is up-to-date with all
        // writes. the stale r_handle is accessed by readers through an Arc clone of atomic pointer
        // inside the MonoReadHandle. oplog contains all the changes that are in w_handle, but not in
        // r_handle.
        //
        // it's now time for us to swap the maps so that readers see up-to-date results from
        // w_handle.

        // prepare w_handle
        let w_handle = self.w_handle.take().unwrap();
        let w_handle = Box::into_raw(w_handle);

        // swap in our w_handle, and get r_handle in return
        let r_handle = self
            .r_handle
            .inner
            .swap(w_handle, atomic::Ordering::Release);
        let r_handle = unsafe { Box::from_raw(r_handle) };

        // ensure that the subsequent epoch reads aren't re-ordered to before the swap
        atomic::fence(atomic::Ordering::SeqCst);

        for (i, epoch) in epochs.iter().enumerate() {
            self.last_epochs[i] = epoch.load(atomic::Ordering::Acquire);
        }

        // NOTE: at this point, there are likely still readers using the w_handle we got
        self.w_handle = Some(r_handle);
        self.second = self.first;
        self.first = false;

        self
    }

    pub fn pending(&self) -> &[MonoOperation<K, V>] {
        &self.oplog[self.swap_index..]
    }

    /// Refresh as necessary to ensure that all operations are visible to readers.
    ///
    /// `MonoWriteHandle::refresh` will *always* wait for old readers to depart and swap the maps.
    /// This method will only do so if there are pending operations.
    pub fn flush(&mut self) -> &mut Self {
        if !self.pending().is_empty() {
            self.refresh();
        }

        self
    }

    /// Set the metadata.
    ///
    /// Will only be visible to readers after the next call to `refresh()`.
    pub fn set_meta(&mut self, mut meta: M) -> M {
        mem::swap(&mut self.meta, &mut meta);
        meta
    }

    fn add_op(&mut self, op: MonoOperation<K, V>) -> &mut Self {
        if !self.first {
            self.oplog.push(op);
        } else {
            // we know there are no outstanding w_handle readers, so we can modify it directly!
            let inner = self.w_handle.as_mut().unwrap();
            Self::apply_second(inner, op);
            // NOTE: since we didn't record this in the oplog, r_handle *must* clone w_handle
        }

        self
    }

    /// Insert the given value at the given key.
    ///
    /// The updated value-set will only be visible to readers after the next call to `refresh()`.
    pub fn insert(&mut self, k: K, v: V) -> &mut Self {
        self.add_op(MonoOperation::Insert(k, v))
    }

    /// Replace the value-set of the given key with the given value.
    ///
    /// With the `smallvec` feature enabled, replacing the value will automatically
    /// deallocate any heap storage and place the new value back into
    /// the `SmallVec` inline storage. This can improve cache locality for common
    /// cases where the value-set is only ever a single element.
    ///
    /// See [the doc section on this](./index.html#small-vector-optimization) for more information.
    ///
    /// The new value will only be visible to readers after the next call to `refresh()`.
    pub fn update<F>(&mut self, k: K, f: F) -> &mut Self
    where
        F: Fn(&mut V) + 'static + Send + Sync,
    {
        self.add_op(MonoOperation::Update(k, Updater(Box::new(f))))
    }

    /// Remove the value-set for the given key.
    ///
    /// The value-set will only disappear from readers after the next call to `refresh()`.
    pub fn remove(&mut self, k: K) -> &mut Self {
        self.add_op(MonoOperation::Remove(k))
    }

    /// Purge all value-sets from the map.
    ///
    /// The map will only appear empty to readers after the next call to `refresh()`.
    ///
    /// Note that this will iterate once over all the keys internally.
    pub fn purge(&mut self) -> &mut Self {
        self.add_op(MonoOperation::Purge)
    }

    /// Apply ops in such a way that no values are dropped, only forgotten
    fn apply_first(inner: &mut Inner<K, V, M, S>, op: &mut MonoOperation<K, V>) {
        match *op {
            MonoOperation::Insert(ref key, ref value) => {
                inner.data.insert(key.clone(), value.clone());
            }

            MonoOperation::Update(ref key, ref updater) => {
                if let Some(val) = inner.data.get_mut(key) {
                    updater.eval(val);
                }
            }
            MonoOperation::Remove(ref key) => {
                #[cfg(not(feature = "indexed"))]
                inner.data.remove(key);
                #[cfg(feature = "indexed")]
                inner.data.swap_remove(key);
            }
            MonoOperation::Purge => {
                inner.data.clear();
            }
        }
    }

    /// Apply operations while allowing dropping of values
    fn apply_second(inner: &mut Inner<K, V, M, S>, op: MonoOperation<K, V>) {
        match op {
            MonoOperation::Insert(key, value) => {
                inner.data.insert(key, value);
            }

            MonoOperation::Update(ref key, ref updater) => {
                if let Some(val) = inner.data.get_mut(key) {
                    updater.eval(val);
                }
            }
            MonoOperation::Remove(ref key) => {
                #[cfg(not(feature = "indexed"))]
                inner.data.remove(key);
                #[cfg(feature = "indexed")]
                inner.data.swap_remove(key);
            }
            MonoOperation::Purge => {
                inner.data.clear();
            }
        }
    }
}

impl<K, V, M, S> Extend<(K, V)> for MonoWriteHandle<K, V, M, S>
where
    K: Eq + Hash + Clone,
    S: BuildHasher + Clone,
    V: Clone,
    M: 'static + Clone,
{
    fn extend<I: IntoIterator<Item = (K, V)>>(&mut self, iter: I) {
        for (k, v) in iter {
            self.insert(k, v);
        }
    }
}

// allow using write handle for reads
use std::ops::Deref;
impl<K, V, M, S> Deref for MonoWriteHandle<K, V, M, S>
where
    K: Eq + Hash + Clone,
    S: BuildHasher + Clone,
    V: Clone,
    M: 'static + Clone,
{
    type Target = MonoReadHandle<K, V, M, S>;
    fn deref(&self) -> &Self::Target {
        &self.r_handle
    }
}
