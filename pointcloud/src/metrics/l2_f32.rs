//! f32 implementations of the L1 metric.

use super::L2;
use crate::base_traits::Metric;
use crate::points::*;
use packed_simd::*;
use std::ops::Deref;

impl Metric<[f32]> for L2 {
    fn dist(x: &[f32], y: &[f32]) -> f32 {
        sq_l2_dense_f32(x.deref(), y.deref()).sqrt()
    }
}

impl<'a> Metric<RawSparse<f32, u32>> for L2 {
    fn dist(x: &RawSparse<f32, u32>, y: &RawSparse<f32, u32>) -> f32 {
        sq_l2_sparse_f32_f32(x.indexes(), x.values(), y.indexes(), y.values()).sqrt()
    }
}

impl<'a> Metric<RawSparse<f32, u16>> for L2 {
    fn dist(x: &RawSparse<f32, u16>, y: &RawSparse<f32, u16>) -> f32 {
        sq_l2_sparse_f32_f32(x.indexes(), x.values(), y.indexes(), y.values()).sqrt()
    }
}

impl<'a> Metric<RawSparse<f32, u8>> for L2 {
    fn dist(x: &RawSparse<f32, u8>, y: &RawSparse<f32, u8>) -> f32 {
        sq_l2_sparse_f32_f32(x.indexes(), x.values(), y.indexes(), y.values()).sqrt()
    }
}

/// basic sparse function
pub fn sq_l2_sparse_f32_f32<S>(x_ind: &[S], x_val: &[f32], y_ind: &[S], y_val: &[f32]) -> f32
where
    S: Ord,
{
    if x_val.is_empty() || y_val.is_empty() {
        if x_val.is_empty() && y_val.is_empty() {
            return 0.0;
        }
        if !x_val.is_empty() && y_val.is_empty() {
            sq_l2_norm_f32(x_val)
        } else {
            sq_l2_norm_f32(y_val)
        }
    } else {
        let mut total = 0.0;
        let (short_iter, mut long_iter) = if x_ind.len() > y_ind.len() {
            (y_ind.iter().zip(y_val), x_ind.iter().zip(x_val))
        } else {
            (x_ind.iter().zip(x_val), y_ind.iter().zip(y_val))
        };

        let mut l_tr: Option<(&S, &f32)> = long_iter.next();
        for (si, sv) in short_iter {
            while let Some((li, lv)) = l_tr {
                if li < si {
                    total += *lv * *lv;
                    l_tr = long_iter.next();
                } else {
                    break;
                }
            }
            if let Some((li, lv)) = l_tr {
                if li == si {
                    let val = *sv - *lv;
                    total += val * val;
                    l_tr = long_iter.next();
                } else {
                    total += *sv * *sv;
                }
            } else {
                total += *sv * *sv;
            }
        }
        while let Some((_li, lv)) = l_tr {
            total += *lv * *lv;
            l_tr = long_iter.next();
        }
        total
    }
}

///
#[inline]
pub fn sq_l2_dense_f32(mut x: &[f32], mut y: &[f32]) -> f32 {
    let mut d_acc_16 = f32x16::splat(0.0);
    while y.len() > 16 {
        let x_simd = f32x16::from_slice_unaligned(x);
        let y_simd = f32x16::from_slice_unaligned(y);
        let diff = x_simd - y_simd;
        d_acc_16 += diff * diff;
        y = &y[16..];
        x = &x[16..];
    }
    let mut d_acc_8 = f32x8::splat(0.0);
    if y.len() > 8 {
        let x_simd = f32x8::from_slice_unaligned(x);
        let y_simd = f32x8::from_slice_unaligned(y);
        let diff = x_simd - y_simd;
        d_acc_8 += diff * diff;
        y = &y[8..];
        x = &x[8..];
    }
    let leftover = y
        .iter()
        .zip(x)
        .map(|(xi, yi)| (xi - yi) * (xi - yi))
        .fold(0.0, |acc, y| acc + y);
    leftover + d_acc_8.sum() + d_acc_16.sum()
}

///
#[inline]
pub fn sq_l2_norm_f32(mut x: &[f32]) -> f32 {
    let mut d_acc_16 = f32x16::splat(0.0);
    while x.len() > 16 {
        let x_simd = f32x16::from_slice_unaligned(x);
        d_acc_16 += x_simd * x_simd;
        x = &x[16..];
    }
    let mut d_acc_8 = f32x8::splat(0.0);
    if x.len() > 8 {
        let x_simd = f32x8::from_slice_unaligned(x);
        d_acc_8 += x_simd * x_simd;
        x = &x[8..];
    }
    let leftover = x.iter().map(|xi| xi * xi).fold(0.0, |acc, xi| acc + xi);
    leftover + d_acc_8.sum() + d_acc_16.sum()
}
