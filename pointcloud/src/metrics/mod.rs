mod l2;
use l2::*;
use std::ops::Deref;

pub trait Metric<T,S>: Send + Sync + 'static
    where
        T: ?Sized,
        S: PartialOrd
    {
    fn dist(x: &T,y: &T) -> S;
    fn norm(x: &T) -> S;
}

#[derive(Debug)]
pub struct L2 {}

pub struct DensePointRef<'a> {
    values: &'a [f32],
}


impl<'a> Deref for DensePointRef<'a> {
    type Target = [f32];
    fn deref(&self) -> &Self::Target {
        &self.values[..]
    }
}


impl<'a, T: AsRef<[f32]>> From<&'a T> for DensePointRef<'a> {
    fn from(arr: &'a T) -> DensePointRef<'a> {
        DensePointRef {
            values: arr.as_ref()
        }
    }
}

impl<'a> Metric<DensePointRef<'a>, f32> for L2 {
    fn dist(x: &DensePointRef,y: &DensePointRef) -> f32 {
        sq_l2_dense_f32(&x[..], &y[..]).sqrt()
    }
    fn norm(x: &DensePointRef) -> f32 {
        sq_l2_norm_f32(&x[..]).sqrt()
    }
}

#[derive(Debug)]
pub struct Sparse<CoefField: std::fmt::Debug, Index: std::fmt::Debug> {
    pub dim: Index,
    pub values: Vec<CoefField>,
    pub indexes: Vec<Index>,
}

impl<'a, CoefField: std::fmt::Debug, Index: std::fmt::Debug + Copy> From<&'a Sparse<CoefField, Index>> for SparseRef<'a, CoefField, Index> {
    fn from(arr: &'a Sparse<CoefField, Index>) -> SparseRef<'a, CoefField, Index> {
        SparseRef {
            dim: arr.dim,
            values: arr.values[..].as_ref(),
            indexes: arr.indexes[..].as_ref(),
        }
    }
}

#[derive(Debug)]
pub struct SparseRef<'a, CoefField: std::fmt::Debug, Index: std::fmt::Debug> {
    pub dim: Index,
    pub values: &'a [CoefField],
    pub indexes: &'a [Index],
}

impl<'a> Metric<SparseRef<'a,f32,u32>, f32> for L2 {
    fn dist(x: &SparseRef<'a,f32,u32>, y: &SparseRef<'a,f32,u32>) -> f32 {
        l2::sq_l2_sparse(&x.indexes[..], x.values, y.indexes, y.values).sqrt()
    }
    fn norm(x: &SparseRef<'a,f32,u32>) -> f32 {
        l2::sq_l2_norm_f32(&x.values).sqrt()
    }
}