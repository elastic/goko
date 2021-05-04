//! # Point Cloud
//! Abstracts data access over several files and glues metadata files to vector data files

use crate::PointRef;
use std::convert::{TryFrom, TryInto};
use std::marker::PhantomData;
use std::ops::Deref;

impl<'a> PointRef for &'a [f32] {
    type DenseIter = std::iter::Copied<std::slice::Iter<'a, f32>>;
    fn dense(&self) -> Vec<f32> {
        Vec::from(*self)
    }
    fn dense_iter(&self) -> Self::DenseIter {
        self.iter().copied()
    }
}

macro_rules! make_misc_point {
    ($base:ident, $iter_name:ident) => {
        /// Helper iterator for converting one type into another. Cleans up a really messy map.
        pub struct $iter_name<'a> {
            iter: std::slice::Iter<'a, $base>,
        }

        impl<'a> Iterator for $iter_name<'a> {
            type Item = f32;
            fn next(&mut self) -> Option<Self::Item> {
                self.iter.next().map(|u| *u as f32)
            }
        }
        impl<'a> PointRef for &'a [$base] {
            type DenseIter = $iter_name<'a>;
            fn dense(&self) -> Vec<f32> {
                self.iter().map(|i| *i as f32).collect()
            }
            fn dense_iter(&self) -> Self::DenseIter {
                $iter_name { iter: self.iter() }
            }
        }
    };
}

make_misc_point!(i8, Converteri8);
make_misc_point!(u8, Converteru8);
make_misc_point!(i16, Converteri16);
make_misc_point!(u16, Converteru16);
make_misc_point!(i32, Converteri32);
make_misc_point!(u32, Converteru32);

#[derive(Debug)]
/// Enables iterating thru a sparse vector, like a dense vector without allocating anything
pub struct SparseDenseIter<'a, T: std::fmt::Debug, S: std::fmt::Debug> {
    raw: RawSparse<T, S>,
    index: usize,
    sparse_index: usize,
    lifetime: PhantomData<&'a T>,
}

impl<'a, T, S> Iterator for SparseDenseIter<'a, T, S>
where
    T: std::fmt::Debug + Copy + Into<f32>,
    S: Ord + TryInto<usize> + std::fmt::Debug + Copy,
{
    type Item = f32;
    fn next(&mut self) -> Option<Self::Item> {
        let dim = self.raw.dim();
        if self.index < dim && self.sparse_index < self.raw.len {
            let raw_si = unsafe { *self.raw.indexes_ptr.add(self.sparse_index) };

            let si: usize = match raw_si.try_into() {
                Ok(si) => si,
                Err(_) => panic!("Could not covert a sparse index into a usize"),
            };

            if si == self.index {
                let val = unsafe { *self.raw.values_ptr.add(self.sparse_index) };
                self.sparse_index += 1;
                self.index += 1;

                Some(val.into())
            } else if self.index < dim {
                self.index += 1;
                Some(0.0)
            } else {
                None
            }
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let dim = self.raw.dim();
        (dim, Some(dim))
    }
}

#[derive(Debug)]
/// The core element of the sparse reference. This has no lifetime information, and you should not build it directly.
/// SparseRef derefences into this, which as the derefrence is borrowed, has a strictly shorter lifespan, and doesn't
/// implement clone or copy, is safe.
pub struct RawSparse<T, S> {
    dim: usize,
    values_ptr: *const T,
    indexes_ptr: *const S,
    len: usize,
}

#[derive(Debug)]
/// A sparse refrence to a pair of buffers, one for indexes and one for the data.
pub struct SparseRef<'a, T, S> {
    raw: RawSparse<T, S>,
    lifetime: PhantomData<&'a T>,
}

unsafe impl<T: Send, S: Send> Send for RawSparse<T, S> {}
unsafe impl<T: Sync, S: Sync> Sync for RawSparse<T, S> {}

impl<T: std::fmt::Debug, S: std::fmt::Debug + TryInto<usize>> RawSparse<T, S> {
    pub(crate) fn indexes<'a>(&'a self) -> &'a [S] {
        unsafe { std::slice::from_raw_parts::<'a>(self.indexes_ptr, self.len) }
    }

    pub(crate) fn values<'a>(&'a self) -> &'a [T] {
        unsafe { std::slice::from_raw_parts::<'a>(self.values_ptr, self.len) }
    }

    pub(crate) fn dim(&self) -> usize {
        self.dim
    }
}

impl<'a, T, S: TryInto<usize>> SparseRef<'a, T, S> {
    /// Creates a new sparse vector reference from a pair of slices.
    pub fn new<'b>(dim: usize, values: &'b [T], indexes: &'b [S]) -> SparseRef<'b, T, S> {
        let len = values.len();
        assert_eq!(
            indexes.len(),
            len,
            "Need the indexes and values to be of identical len"
        );
        let indexes_ptr: *const S = indexes.as_ptr();
        let values_ptr: *const T = values.as_ptr();
        let raw = RawSparse {
            indexes_ptr,
            values_ptr,
            dim,
            len,
        };
        SparseRef {
            raw,
            lifetime: PhantomData,
        }
    }

    /// The underlying indexes.
    pub fn indexes(&self) -> &'a [S] {
        unsafe { std::slice::from_raw_parts::<'a>(self.raw.indexes_ptr, self.raw.len) }
    }

    /// The underlying values.
    pub fn values(&self) -> &'a [T] {
        unsafe { std::slice::from_raw_parts::<'a>(self.raw.values_ptr, self.raw.len) }
    }

    /// The dimension of this point.
    pub fn dim(&self) -> usize {
        self.raw.dim
    }
}

impl<'a, T, S> Deref for SparseRef<'a, T, S> {
    type Target = RawSparse<T, S>;
    fn deref(&self) -> &Self::Target {
        &self.raw
    }
}

impl<'a, T, S> PointRef for SparseRef<'a, T, S>
where
    T: std::fmt::Debug + Copy + Into<f32> + Send + Sync,
    S: TryInto<usize> + Ord + TryFrom<usize> + std::fmt::Debug + Copy + Send + Sync + 'static,
{
    type DenseIter = SparseDenseIter<'a, T, S>;

    fn dense(&self) -> Vec<f32> {
        let dim = self.dim();
        let mut v = vec![0.0; dim];

        for (xi, i) in self.values().iter().zip(self.indexes()) {
            match (*i).try_into() {
                Ok(i) => {
                    let _index: usize = i;
                    v[i] = (*xi).into();
                }
                Err(_) => panic!("Could not covert a sparse index into a usize"),
            }
        }
        v
    }

    fn dense_iter(&self) -> SparseDenseIter<'a, T, S> {
        let raw = RawSparse {
            dim: self.raw.dim,
            values_ptr: self.raw.values_ptr,
            indexes_ptr: self.raw.indexes_ptr,
            len: self.raw.len,
        };
        SparseDenseIter {
            raw,
            index: 0,
            sparse_index: 0,
            lifetime: PhantomData,
        }
    }
}
