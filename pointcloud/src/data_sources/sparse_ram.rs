
use crate::pc_errors::{PointCloudError, PointCloudResult};
use std::marker::PhantomData;
use std::convert::{TryInto, TryFrom};

use crate::metrics::{SparseRef,L2};

use crate::base_traits::*;
use crate::label_sources::VecLabels;

/// The data stored in ram.
#[derive(Debug)]
pub struct SparseDataRam<CoefField: std::fmt::Debug = f32, Index: std::fmt::Debug = u32, M = L2> {
    name: String,
    values: Vec<CoefField>,
    col_index: Vec<Index>,
    row_index: Vec<Index>,
    dim: usize,
    metric: PhantomData<M>,
}

impl<CoefField, Index> SparseDataRam<CoefField, Index>
 where
 CoefField: std::fmt::Debug + 'static,
 Index: std::fmt::Debug + 'static {
    pub fn new(values: Vec<CoefField>, col_index: Vec<Index>, row_index: Vec<Index>, dim: usize) -> SparseDataRam<CoefField, Index> {
        SparseDataRam::<CoefField, Index> {
            name: String::new(),
            values,
            col_index,
            row_index,
            dim,
            metric: PhantomData,
        }
    }
}


impl PointCloud for SparseDataRam {
    type PointRef = SparseRef<'a,f32,u32>;
    type Field = f32;
    type Metric = L2;

    /// The number of samples this cloud covers
    fn len(&self) -> usize {
        self.row_index.len() - 1
    }
    /// If this is empty
    fn is_empty(&self) -> bool {
        self.row_index.len() > 1
    }
    /// The dimension of the underlying data
    fn dim(&self) -> usize {
        self.dim
    }
    /// Indexes used for access
    fn reference_indexes(&self) -> Vec<usize> {
       (0..self.len()).collect()
    }
    /// Gets a point from this dataset
    fn point(&'a self, pn: usize) -> PointCloudResult<SparseRef<'a,f32,u32>> {
        let lower_bound = self.row_index[pn].try_into();
        let upper_bound = self.row_index[pn + 1].try_into();
        let dim = self.dim.try_into();
        if let (Ok(lower_bound),Ok(upper_bound),Ok(dim)) = (lower_bound,upper_bound,dim) {
            let values = &self.values[lower_bound..upper_bound];
            let indexes = &self.col_index[lower_bound..upper_bound];
            Ok(SparseRef {
                dim,
                values,
                indexes,
            })
        } else {
            panic!("Could not covert a usize into a sparse dimension");
        }
    }
}
