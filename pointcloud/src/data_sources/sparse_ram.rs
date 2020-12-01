
use crate::pc_errors::PointCloudResult;
use std::marker::PhantomData;
use std::convert::TryInto;

use crate::points::*;
use crate::metrics::*;

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

impl<CoefField, Index, M> SparseDataRam<CoefField, Index, M>
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


impl<M> PointCloud<RawSparse<f32, u32>> for SparseDataRam<f32,u32, M> 
where 
    M: Metric<RawSparse<f32, u32>,f32>,
{
    type PointRef<'a> = SparseRef<'a, f32, u32>;
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
    fn point<'a,'b: 'a>(&'b self, pn: usize) -> PointCloudResult<Self::PointRef<'a>> {
        let lower_bound = self.row_index[pn].try_into();
        let upper_bound = self.row_index[pn + 1].try_into();
        if let (Ok(lower_bound),Ok(upper_bound)) = (lower_bound,upper_bound) {
            let values = &self.values[lower_bound..upper_bound];
            let indexes = &self.col_index[lower_bound..upper_bound];
            Ok(SparseRef::new(self.dim,values,indexes))
        } else {
            panic!("Could not covert a usize into a sparse dimension");
        }
    }
}