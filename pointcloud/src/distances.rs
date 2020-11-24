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

//! Supported distances

//use super::PointRef;
use crate::pc_errors::*;
use packed_simd::*;
use std::fmt::Debug;



/// The trait that enables a metric
pub trait Metric: 'static + Send + Sync + Debug + Clone {
    /// Dense calculation
    fn dense(x: &[f32], y: &[f32]) -> f32;
    /// Sparse calculation, we assume that the index slices are in accending order and
    /// that the values correspond to the indexes
    fn sparse(x_ind: &[u32], x_val: &[f32], y_ind: &[u32], y_val: &[f32]) -> f32;
    /// The norm, dense(x,x)
    fn norm(x: &[f32]) -> f32;
    /// Useful external calculation
    fn dist<'a, 'b, T, S>(x: T, y: S) -> PointCloudResult<f32>
    where
        T: Into<PointRef<'a>>,
        S: Into<PointRef<'b>>,
    {
        match ((x).into(), (y).into()) {
            (PointRef::Dense(x_vals), PointRef::Dense(y_vals)) => Ok((Self::dense)(x_vals, y_vals)),
            (PointRef::Sparse(x_vals, x_ind), PointRef::Sparse(y_vals, y_inds)) => {
                Ok((Self::sparse)(x_ind, x_vals, y_inds, y_vals))
            }
            _ => Err(PointCloudError::MetricError),
        }
    }
}

/// L2 norm, the square root of the sum of squares
#[derive(Debug, Clone)]
pub struct L2 {}

/*
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;
#[cfg(target_arch = "x86")]
use std::arch::x86::*;

fn dense_l2_safe(x: &[f32], y: &[f32]) -> f32 {
    y.iter().zip(x).map(|(xi, yi)| (xi - yi) * (xi - yi)).fold(0.0, |acc, y| acc + y)
}

fn dense_l2_avx(mut x: &[f32], mut y: &[f32]) -> f32 {
    assert_eq!(x.len(),y.len());
    let mut index = 0;
    let len = x.len();
    let mut x_ptr = x.as_ptr();
    let mut y_ptr = y.as_ptr();

    let mut avx_acc = unsafe {_mm256_setzero_ps()};
    while len - index > 8 {
        unsafe {
            let x_simd = _mm256_loadu_ps(x_ptr);
            let y_simd = _mm256_loadu_ps(y_ptr);
            let diff = _mm256_sub_ps(x_simd,y_simd);
            avx_acc = _mm256_fmadd_ps(diff,diff,avx_acc);

            x_ptr = x_ptr.offset(8);
            y_ptr = y_ptr.offset(8);
        }
        index += 8;
    }

    let simd: [f32;8] = unsafe { std::mem::transmute(avx_acc) };

    y = &y[index..];
    x = &x[index..];
    let leftover: f32 = y
        .iter()
        .zip(x)
        .map(|(xi, yi)| (xi - yi) * (xi - yi))
        .fold(0.0, |acc, y| acc + y);

    simd.iter().fold(leftover,|a,x| a+x).sqrt()
}


use simdeez::*;
use simdeez::scalar::*;
use simdeez::sse2::*;
use simdeez::sse41::*;
use simdeez::avx::*;
use simdeez::avx2::*;
// If you want your SIMD function to use use runtime feature detection to call
// the fastest available version, use the simd_runtime_generate macro:
simd_runtime_generate!(
fn l2_distance(x: &[f32], y: &[f32]) -> Vec<f32> {
    
    /// Set each slice to the same length for iteration efficiency
    let mut x = &x[..x.len()];
    let mut y = &y[..x.len()];
    let mut acc = S::setzero_ps();
    let mut leftover: f32 = 0;

    // Operations have to be done in terms of the vector width
    // so that it will work with any size vector.
    // the width of a vector type is provided as a constant
    // so the compiler is free to optimize it more.
    // S::VF32_WIDTH is a constant, 4 when using SSE, 8 when using AVX2, etc
    while x.len() >= S::VF32_WIDTH {
        //load data from your vec into an SIMD value
        let xv = S::loadu_ps(&x[0]);
        let yv = S::loadu_ps(&y[0]);

        // Use the usual intrinsic syntax if you prefer
        let mut diff = S::sub_ps(xv1, yv1);
        acc += diff*diff;
        
        // Move each slice to the next position
        x = &x[S::VF32_WIDTH..];
        y = &y[S::VF32_WIDTH..];
    }
    
    // (Optional) Compute the remaining elements. Not necessary if you are sure the length
    // of your data is always a multiple of the maximum S::VF32_WIDTH you compile for (4 for SSE, 8 for AVX2, etc).
    // This can be asserted by putting `assert_eq!(x1.len(), 0);` here
    for i in 0..x1.len() {
        let diff = x[i] - y[i];
        leftover += diff*diff;
    }
    for i in 0..S::VF32_WIDTH {
        leftover += acc[i];
    }
    leftover.sqrt()
});
*/

impl Metric for L2 {
    #[inline]
    fn dense(mut x: &[f32], mut y: &[f32]) -> f32 {
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
        (leftover + d_acc_8.sum() + d_acc_16.sum()).sqrt()
    }

    #[inline]
    fn norm(mut x: &[f32]) -> f32 {
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
        (leftover + d_acc_8.sum() + d_acc_16.sum()).sqrt()
    }

    fn sparse(x_ind: &[u32], x_val: &[f32], y_ind: &[u32], y_val: &[f32]) -> f32 {
        if x_val.is_empty() || y_val.is_empty() {
            if x_val.is_empty() && y_val.is_empty() {
                return 0.0;
            }
            if !x_val.is_empty() && y_val.is_empty() {
                Self::norm(x_val)
            } else {
                Self::norm(y_val)
            }
        } else {
            let mut total = 0.0;
            let (short_iter, mut long_iter) = if x_ind.len() > y_ind.len() {
                (y_ind.iter().zip(y_val), x_ind.iter().zip(x_val))
            } else {
                (x_ind.iter().zip(x_val), y_ind.iter().zip(y_val))
            };

            let mut l_tr: Option<(&u32, &f32)> = long_iter.next();
            for (si, sv) in short_iter {
                while let Some((li, lv)) = l_tr {
                    if li < si {
                        total += lv * lv;
                        l_tr = long_iter.next();
                    } else {
                        break;
                    }
                }
                if let Some((li, lv)) = l_tr {
                    if li == si {
                        let val = sv - lv;
                        total += val * val;
                        l_tr = long_iter.next();
                    } else {
                        total += sv * sv;
                    }
                } else {
                    total += sv * sv;
                }
            }
            while let Some((_li, lv)) = l_tr {
                total += lv * lv;
                l_tr = long_iter.next();
            }
            total.sqrt()
        }
    }
}

/// L infity norm, the max of the absolute values of the elements
#[derive(Debug, Clone)]
pub struct Linfty {}

impl Metric for Linfty {
    #[inline]
    fn dense(mut x: &[f32], mut y: &[f32]) -> f32 {
        let mut d_acc_16 = f32x16::splat(0.0);
        while y.len() > 16 {
            let y_simd = f32x16::from_slice_unaligned(y);
            let x_simd = f32x16::from_slice_unaligned(x);
            let diff = (x_simd - y_simd).abs();
            d_acc_16 = d_acc_16.max(diff);
            y = &y[16..];
            x = &x[16..];
        }
        let mut d_acc_8 = f32x8::splat(0.0);
        if y.len() > 8 {
            let y_simd = f32x8::from_slice_unaligned(y);
            let x_simd = f32x8::from_slice_unaligned(x);
            let diff = (x_simd - y_simd).abs();
            d_acc_8 = d_acc_8.max(diff);
            y = &y[8..];
            x = &x[8..];
        }
        let leftover = y
            .iter()
            .zip(x)
            .map(|(xi, yi)| (xi - yi).abs())
            .fold(0.0, |acc: f32, y| acc.max(y));
        leftover.max(d_acc_8.max_element().max(d_acc_16.max_element()))
    }

    #[inline]
    fn norm(mut x: &[f32]) -> f32 {
        let mut d_acc_16 = f32x16::splat(0.0);
        while x.len() > 16 {
            let x_simd = f32x16::from_slice_unaligned(x);
            d_acc_16 = d_acc_16.max(x_simd);
            x = &x[16..];
        }
        let mut d_acc_8 = f32x8::splat(0.0);
        if x.len() > 8 {
            let x_simd = f32x8::from_slice_unaligned(x);
            d_acc_8 = d_acc_8.max(x_simd);
            x = &x[8..];
        }
        let leftover = x
            .iter()
            .map(|xi| xi.abs())
            .fold(0.0, |acc: f32, xi| acc.max(xi));
        leftover.max(d_acc_8.max_element().max(d_acc_16.max_element()))
    }

    fn sparse(x_ind: &[u32], x_val: &[f32], y_ind: &[u32], y_val: &[f32]) -> f32 {
        if x_val.is_empty() || y_val.is_empty() {
            if x_val.is_empty() && y_val.is_empty() {
                return 0.0;
            }
            if !x_val.is_empty() && y_val.is_empty() {
                Self::norm(x_val)
            } else {
                Self::norm(y_val)
            }
        } else {
            let mut max_val: f32 = 0.0;
            let (short_iter, mut long_iter) = if x_ind.len() > y_ind.len() {
                (y_ind.iter().zip(y_val), x_ind.iter().zip(x_val))
            } else {
                (x_ind.iter().zip(x_val), y_ind.iter().zip(y_val))
            };

            let mut l_tr: Option<(&u32, &f32)> = long_iter.next();
            for (si, sv) in short_iter {
                while let Some((li, lv)) = l_tr {
                    if li < si {
                        max_val = max_val.max(lv * lv);
                        l_tr = long_iter.next();
                    } else {
                        break;
                    }
                }
                if let Some((li, lv)) = l_tr {
                    if li == si {
                        let val = sv - lv;
                        max_val = max_val.max(val * val);
                        l_tr = long_iter.next();
                    } else {
                        max_val = max_val.max(sv * sv);
                    }
                } else {
                    max_val = max_val.max(sv * sv);
                }
            }
            while let Some((_li, lv)) = l_tr {
                max_val = max_val.max(lv * lv);
                l_tr = long_iter.next();
            }
            max_val.sqrt()
        }
    }
}

/// L1 norm, the sum of absolute values
#[derive(Debug, Clone)]
pub struct L1 {}

impl Metric for L1 {
    #[inline]
    fn dense(mut x: &[f32], mut y: &[f32]) -> f32 {
        let mut d_acc_16 = f32x16::splat(0.0);
        while y.len() > 16 {
            let y_simd = f32x16::from_slice_unaligned(y);
            let x_simd = f32x16::from_slice_unaligned(x);
            let diff = x_simd - y_simd;
            d_acc_16 += diff.abs();
            y = &y[16..];
            x = &x[16..];
        }
        let mut d_acc_8 = f32x8::splat(0.0);
        if y.len() > 8 {
            let y_simd = f32x8::from_slice_unaligned(y);
            let x_simd = f32x8::from_slice_unaligned(x);
            let diff = x_simd - y_simd;
            d_acc_8 += diff.abs();
            y = &y[8..];
            x = &x[8..];
        }
        let leftover = y
            .iter()
            .zip(x)
            .map(|(xi, yi)| (xi - yi).abs())
            .fold(0.0, |acc, y| acc + y);
        leftover + d_acc_8.sum() + d_acc_16.sum()
    }

    #[inline]
    fn norm(mut x: &[f32]) -> f32 {
        let mut d_acc_16 = f32x16::splat(0.0);
        while x.len() > 16 {
            let x_simd = f32x16::from_slice_unaligned(x);
            d_acc_16 += x_simd.abs();
            x = &x[16..];
        }
        let mut d_acc_8 = f32x8::splat(0.0);
        if x.len() > 8 {
            let x_simd = f32x8::from_slice_unaligned(x);
            d_acc_8 += x_simd.abs();
            x = &x[8..];
        }
        let leftover = x.iter().map(|xi| xi.abs()).fold(0.0, |acc, xi| acc + xi);
        leftover + d_acc_8.sum() + d_acc_16.sum()
    }

    fn sparse(x_ind: &[u32], x_val: &[f32], y_ind: &[u32], y_val: &[f32]) -> f32 {
        if x_val.is_empty() || y_val.is_empty() {
            if x_val.is_empty() && y_val.is_empty() {
                return 0.0;
            }
            if !x_val.is_empty() && y_val.is_empty() {
                Self::norm(x_val)
            } else {
                Self::norm(y_val)
            }
        } else {
            let mut total = 0.0;
            let (short_iter, mut long_iter) = if x_ind.len() > y_ind.len() {
                (y_ind.iter().zip(y_val), x_ind.iter().zip(x_val))
            } else {
                (x_ind.iter().zip(x_val), y_ind.iter().zip(y_val))
            };

            let mut l_tr: Option<(&u32, &f32)> = long_iter.next();
            for (si, sv) in short_iter {
                while let Some((li, lv)) = l_tr {
                    if li < si {
                        total += lv.abs();
                        l_tr = long_iter.next();
                    } else {
                        break;
                    }
                }
                if let Some((li, lv)) = l_tr {
                    if li == si {
                        let val = sv - lv;
                        total += val.abs();
                        l_tr = long_iter.next();
                    } else {
                        total += sv.abs();
                    }
                } else {
                    total += sv.abs();
                }
            }
            while let Some((_li, lv)) = l_tr {
                total += lv.abs();
                l_tr = long_iter.next();
            }
            total
        }
    }
}

/// Not a norm! Still, helpful for document clouds and the like
#[derive(Debug, Clone)]
pub struct CosineSim {}

impl Metric for CosineSim {
    #[inline]
    fn dense(mut x: &[f32], mut y: &[f32]) -> f32 {
        let mut d_acc_16 = f32x16::splat(0.0);
        let mut x_acc_16 = f32x16::splat(0.0);
        let mut y_acc_16 = f32x16::splat(0.0);
        while y.len() > 16 {
            let y_simd = f32x16::from_slice_unaligned(y);
            let x_simd = f32x16::from_slice_unaligned(x);
            d_acc_16 += x_simd * y_simd;
            x_acc_16 += x_simd * x_simd;
            y_acc_16 += y_simd * y_simd;
            y = &y[16..];
            x = &x[16..];
        }
        let mut d_acc_8 = f32x8::splat(0.0);
        let mut x_acc_8 = f32x8::splat(0.0);
        let mut y_acc_8 = f32x8::splat(0.0);
        if y.len() > 8 {
            let y_simd = f32x8::from_slice_unaligned(y);
            let x_simd = f32x8::from_slice_unaligned(x);
            d_acc_8 += x_simd * y_simd;
            x_acc_8 += x_simd * x_simd;
            y_acc_8 += y_simd * y_simd;
            y = &y[8..];
            x = &x[8..];
        }
        let acc_leftover = y
            .iter()
            .zip(x)
            .map(|(xi, yi)| xi * yi)
            .fold(0.0, |acc, y| acc + y);
        let y_leftover = y.iter().map(|yi| yi * yi).fold(0.0, |acc, yi| acc + yi);
        let x_leftover = x.iter().map(|xi| xi * xi).fold(0.0, |acc, xi| acc + xi);
        let acc = acc_leftover + d_acc_8.sum() + d_acc_16.sum();
        let xnm = (x_leftover + x_acc_8.sum() + x_acc_16.sum()).sqrt();
        let ynm = (y_leftover + y_acc_8.sum() + y_acc_16.sum()).sqrt();
        acc / (xnm * ynm).max(0.00001)
    }

    fn norm(_x: &[f32]) -> f32 {
        0.0
    }

    fn sparse(x_ind: &[u32], x_val: &[f32], y_ind: &[u32], y_val: &[f32]) -> f32 {
        if x_val.is_empty() || y_val.is_empty() {
            0.0
        } else {
            let mut dotprod = 0.0;
            let (short_iter, mut long_iter) = if x_ind.len() > y_ind.len() {
                (y_ind.iter().zip(y_val), x_ind.iter().zip(x_val))
            } else {
                (x_ind.iter().zip(x_val), y_ind.iter().zip(y_val))
            };

            let mut l_tr: Option<(&u32, &f32)> = long_iter.next();
            for (si, sv) in short_iter {
                while let Some((_li, _lv)) = l_tr {
                    l_tr = long_iter.next();
                }
                if let Some((li, lv)) = l_tr {
                    if li == si {
                        dotprod += sv * lv;
                        l_tr = long_iter.next();
                    }
                }
            }
            let xnm = L2::norm(x_val);
            let ynm = L2::norm(y_val);
            dotprod / (xnm * ynm).max(0.00001)
        }
    }
}
