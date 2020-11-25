//! # Point Cloud
//! Abstracts data access over several files and glues metadata files to vector data files

use crate::{distances, Point};

pub mod sparse;

#[derive(Debug)]
pub struct DenseL2<'a, T> {
    pub values: &'a [T],
}


impl<'a> Point<'a, f32> for DenseL2<'a, f32> {
    type DenseIter =  std::iter::Cloned<std::slice::Iter<'a,f32>>;

    fn dist(&self,other: &Self) -> f32 {
        distances::l2::sq_l2_dense_f32(&self.values[..],&other.values[..]).sqrt()
    }
    fn norm(&self) -> f32 {
        distances::l2::sq_l2_norm_f32(&self.values[..]).sqrt()
    }
    fn dense(&self) -> Vec<f32> {
        Vec::from(self.values)
    }
    fn dense_iter(&self) -> Self::DenseIter {
        self.values.iter().cloned()
    }
}

impl<'a> From<&'a [f32]> for DenseL2<'a, f32> {
    fn from(arr: &'a [f32]) -> DenseL2<'a, f32> {
        DenseL2 {
            values: arr,
        }
    }
}

impl<'a, T: AsRef<[f32]>> From<&'a T> for DenseL2<'a, f32> {
    fn from(arr: &'a T) -> DenseL2<'a, f32> {
        DenseL2 {
            values: arr.as_ref(),
        }
    }
}

impl<'a> Point<'a, f64> for DenseL2<'a, f64> {
    type DenseIter =  std::iter::Cloned<std::slice::Iter<'a,f64>>;

    fn dist(&self,other: &Self) -> f64 {
        distances::l2::sq_l2_dense_f64(&self.values[..],&other.values[..]).sqrt()
    }
    fn norm(&self) -> f64 {
        distances::l2::sq_l2_norm_f64(&self.values[..]).sqrt()
    }
    fn dense(&self) -> Vec<f64> {
        self.values.iter().map(|x| *x).collect()
    }
    fn dense_iter(&self) -> Self::DenseIter {
        self.values.iter().cloned()
    }
}