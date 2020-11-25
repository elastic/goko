use crate::{distances, Point};
use std::convert::{TryInto, TryFrom};

#[derive(Debug)]
/// Enables iterating thru a sparse vector, like a dense vector without allocating anythin
pub struct SparseDenseIter<'a, T: std::fmt::Debug, S: std::fmt::Debug> {
    sparse: SparseL2<'a, T, S>,
    index: usize,
    sparse_index: usize,
    dim: usize,
}

impl<'a, T: std::fmt::Debug, S: std::fmt::Debug> Iterator for SparseDenseIter<'a, T, S> 
    where
    T: Default + Clone + Copy,
    S: Ord + TryInto<usize> + std::fmt::Debug + Clone + Copy  {
    type Item = T;
    fn next(&mut self) -> Option<Self::Item> {
        if self.index < self.dim {
            match self.sparse.indexes[self.sparse_index].try_into() {
                Ok(si) => {
                    if si == self.index  {
                        self.sparse_index += 1;
                        self.index += 1;
                        Some(self.sparse.values[self.sparse_index - 1])
                    } else if self.index < self.dim {
                        self.index += 1;
                        Some(T::default())
                    } else {
                        None
                    }
                }
                Err(_) => panic!("Could not covert a sparse index into a usize"),
            }
            
        } else {
            None
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.dim, Some(self.dim))
    }
}

#[derive(Debug)]
struct SparseL2<'a, T: std::fmt::Debug, S: std::fmt::Debug> {
    pub dim: S,
    pub values: &'a [T],
    pub indexes: &'a [S],
}

impl<'a, S: TryInto<usize> + Ord + TryFrom<usize> + std::fmt::Debug + Clone + Copy> Point<'a, f32> for SparseL2<'a, f32, S> {
    type DenseIter = SparseDenseIter<'a, f32, S>;
    fn dist(&self,other: &Self) -> f32 {
        distances::l2::sq_l2_sparse(self.indexes, self.values, other.indexes, other.values).sqrt()
    }
    fn norm(&self) -> f32 {
        distances::l2::sq_l2_norm_f32(&self.values).sqrt()
    }
    fn dense(&self) -> Vec<f32> {
        match self.dim.try_into() {
            Ok(dim) => {
                let mut v = vec![0.0;dim];
                for (xi,i) in self.values.iter().zip(self.indexes) {
                    match (*i).try_into() {
                        Ok(i) => {
                            let _index: usize = i; 
                            v[i] = *xi;
                        }
                        Err(_) => panic!("Could not covert a sparse index into a usize"),
                    }
                }
                v
            }
            Err(_) => panic!("Could not covert a sparse dimension into a usize"),
        }
    }

    fn dense_iter(&self) -> SparseDenseIter<'a, f32, S> {
        match self.dim.try_into() {
            Ok(dim) => {
                let sparse = SparseL2 {
                    dim: self.dim,
                    values: &self.values[..],
                    indexes: &self.indexes[..],
                };
                SparseDenseIter {
                    sparse: sparse,
                    index: 0,
                    sparse_index: 0,
                    dim,
                }
            },
            Err(_) => panic!("Could not covert a sparse dimension into a usize"),
        }
        
    }
}

impl<'a, S: TryInto<usize> + Ord + TryFrom<usize> + std::fmt::Debug + Clone + Copy> Point<'a,f64> for SparseL2<'a, f64, S> {
    type DenseIter = SparseDenseIter<'a, f64, S>;
    fn dist(&self,other: &Self) -> f64 {
        distances::l2::sq_l2_sparse(self.indexes, self.values, other.indexes, other.values).sqrt()
    }
    fn norm(&self) -> f64 {
        distances::l2::sq_l2_norm_f64(&self.values).sqrt()
    }
    fn dense(&self) -> Vec<f64> {
        match self.dim.try_into() {
            Ok(dim) => {
                let mut v = vec![0.0;dim];
                for (xi,i) in self.values.iter().zip(self.indexes) {
                    match (*i).try_into() {
                        Ok(i) => {
                            let index: usize = i; 
                            v[i] = *xi;
                        }
                        Err(_) => panic!("Could not covert a sparse index into a usize"),
                    }
                }
                v
            }
            Err(_) => panic!("Could not covert a sparse dimension into a usize"),
        }
    }

    fn dense_iter(&self) -> SparseDenseIter<'a, f64, S> {
        match self.dim.try_into() {
            Ok(dim) => {
                let sparse = SparseL2 {
                    dim: self.dim,
                    values: &self.values[..],
                    indexes: &self.indexes[..],
                };
                SparseDenseIter {
                    sparse: sparse,
                    index: 0,
                    sparse_index: 0,
                    dim,
                }
            },
            Err(_) => panic!("Could not covert a sparse dimension into a usize"),
        }
        
    }
}