use crate::base_traits::Metric;
use std::ops::Deref;
use crate::points::*;
use packed_simd::*;
use super::L1;

macro_rules! make_l1_distance {
    ($base:ident, $simd_16_base:ident, $simd_8_base:ident, $sparse_base:ident, $dist_base:ident, $norm_base:ident) => {
        ///
        #[inline]
        pub fn $dist_base(mut x: &[$base], mut y: &[$base]) -> f32 {
            let mut d_acc_16 = f32x16::splat(0.0);
            while y.len() > 16 {
                let x_simd = $simd_16_base::from_slice_unaligned(x);
                let y_simd = $simd_16_base::from_slice_unaligned(y);
                let x_simd_f32 = f32x16::from_cast(x_simd);
                let y_simd_f32 = f32x16::from_cast(y_simd);
                let diff = x_simd_f32 - y_simd_f32;
                d_acc_16 += diff.abs();
                y = &y[16..];
                x = &x[16..];
            }
            let mut d_acc_8 = f32x8::splat(0.0);
            if y.len() > 8 {
                let x_simd = $simd_8_base::from_slice_unaligned(x);
                let y_simd = $simd_8_base::from_slice_unaligned(y);
                let x_simd_f32 = f32x8::from_cast(x_simd);
                let y_simd_f32 = f32x8::from_cast(y_simd);
                let diff = x_simd_f32 - y_simd_f32;
                d_acc_8 += diff.abs();
                y = &y[8..];
                x = &x[8..];
            }
            let leftover = y
                .iter()
                .zip(x)
                .map(|(xi, yi)| (*xi as f32 - *yi as f32).abs())
                .fold(0.0, |acc, y| acc + y);
            leftover + d_acc_8.sum() + d_acc_16.sum()
        }

        ///
        #[inline]
        pub fn $norm_base(mut x: &[$base]) -> f32 {
            let mut d_acc_16 = f32x16::splat(0.0);
            while x.len() > 16 {
                let x_simd = $simd_16_base::from_slice_unaligned(x);
                let x_simd_f32 = f32x16::from_cast(x_simd);
                d_acc_16 += x_simd_f32.abs();
                x = &x[16..];
            }
            let mut d_acc_8 = f32x8::splat(0.0);
            if x.len() > 8 {
                let x_simd = $simd_8_base::from_slice_unaligned(x);
                let x_simd_f32 = f32x8::from_cast(x_simd);
                d_acc_8 += x_simd_f32.abs();
                x = &x[8..];
            }
            let leftover = x
                .iter()
                .map(|xi| (*xi as f32).abs())
                .fold(0.0, |acc, y| acc + y);
            leftover + d_acc_8.sum() + d_acc_16.sum()
        }

        /// basic sparse function
        pub fn $sparse_base<S>(x_ind: &[S], x_val: &[$base], y_ind: &[S], y_val: &[$base]) -> f32
        where
            S: Ord,
        {
            if x_val.is_empty() || y_val.is_empty() {
                if x_val.is_empty() && y_val.is_empty() {
                    return 0.0;
                }
                if !x_val.is_empty() && y_val.is_empty() {
                    $norm_base(x_val)
                } else {
                    $norm_base(y_val)
                }
            } else {
                let mut total = 0.0;
                let (short_iter, mut long_iter) = if x_ind.len() > y_ind.len() {
                    (y_ind.iter().zip(y_val), x_ind.iter().zip(x_val))
                } else {
                    (x_ind.iter().zip(x_val), y_ind.iter().zip(y_val))
                };

                let mut l_tr: Option<(&S, &$base)> = long_iter.next();
                for (si, sv) in short_iter {
                    while let Some((li, lv)) = l_tr {
                        if li < si {
                            total += (*lv as f32).abs();
                            l_tr = long_iter.next();
                        } else {
                            break;
                        }
                    }
                    if let Some((li, lv)) = l_tr {
                        if li == si {
                            let val = (*sv as f32) - (*lv as f32);
                            total += val.abs();
                            l_tr = long_iter.next();
                        } else {
                            total += (*sv as f32).abs();
                        }
                    } else {
                        total += (*sv as f32).abs();
                    }
                }
                while let Some((_li, lv)) = l_tr {
                    total += (*lv as f32).abs();
                    l_tr = long_iter.next();
                }
                total
            }
        }
        impl Metric<[$base]> for L1 {
            fn dist(x: &[$base], y: &[$base]) -> f32 {
                $dist_base(x.deref(), y.deref()).sqrt()
            }
        }
        
        impl<'a> Metric<RawSparse<$base, u32>> for L1 {
            fn dist(x: &RawSparse<$base, u32>, y: &RawSparse<$base, u32>) -> f32 {
                $sparse_base(x.indexes(), x.values(), y.indexes(), y.values()).sqrt()
            }
        }
        
        impl<'a> Metric<RawSparse<$base, u16>> for L1 {
            fn dist(x: &RawSparse<$base, u16>, y: &RawSparse<$base, u16>) -> f32 {
                $sparse_base(x.indexes(), x.values(), y.indexes(), y.values()).sqrt()
            }
        }
        
        impl<'a> Metric<RawSparse<$base, u8>> for L1 {
            fn dist(x: &RawSparse<$base, u8>, y: &RawSparse<$base, u8>) -> f32 {
                $sparse_base(x.indexes(), x.values(), y.indexes(), y.values()).sqrt()
            }
        }
    }
}

make_l1_distance!(i8,i8x16,i8x8,l1_sparse_i8_f32,l1_dist_i8,l1_norm_i8);
make_l1_distance!(u8,u8x16,u8x8,l1_sparse_u8_f32,l1_dist_u8,l1_norm_u8);
make_l1_distance!(i16,i16x16,i16x8,l1_sparse_i16_f32,l1_dist_i16,l1_norm_i16);
make_l1_distance!(u16,u16x16,u16x8,l1_sparse_u16_f32,l1_dist_u16,l1_norm_u16);
make_l1_distance!(i32,i32x16,i32x8,l1_sparse_i32_f32,l1_dist_i32,l1_norm_i32);
make_l1_distance!(u32,u32x16,u32x8,l1_sparse_u32_f32,l1_dist_u32,l1_norm_u32);
