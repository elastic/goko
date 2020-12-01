mod l2_raw;
use crate::base_traits::Metric;
use l2_raw::*;
use std::ops::Deref;

use crate::points::*;

#[derive(Debug)]
pub struct L2 {}

impl Metric<[f32]> for L2 {
    fn dist(x: &[f32], y: &[f32]) -> f32 {
        sq_l2_dense_f32(x.deref(), y.deref()).sqrt()
    }
    fn norm(x: &[f32]) -> f32 {
        sq_l2_norm_f32(x.deref()).sqrt()
    }
}

impl<'a> Metric<RawSparse<f32, u32>> for L2 {
    fn dist(x: &RawSparse<f32, u32>, y: &RawSparse<f32, u32>) -> f32 {
        sq_l2_sparse(x.indexes(), x.values(), y.indexes(), y.values()).sqrt()
    }
    fn norm(x: &RawSparse<f32, u32>) -> f32 {
        sq_l2_norm_f32(x.values()).sqrt()
    }
}
