use crate::pc_errors::ParsingError;
use crate::pc_errors::PointCloudResult;
use std::convert::TryInto;
use std::marker::PhantomData;

use crate::base_traits::*;
use crate::metrics::*;
use crate::points::*;

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
    Index: std::fmt::Debug + 'static,
{
    pub fn new(
        values: Vec<CoefField>,
        col_index: Vec<Index>,
        row_index: Vec<Index>,
        dim: usize,
    ) -> SparseDataRam<CoefField, Index> {
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

impl<M> PointCloud for SparseDataRam<f32, u32, M>
where
    M: Metric<RawSparse<f32, u32>>,
{
    type PointRef<'a> = SparseRef<'a, f32, u32>;
    type Point = RawSparse<f32, u32>;
    type Metric = L2;
    type LabelSummary = ();
    type Label = ();
    type MetaSummary = ();
    type Metadata = ();

    fn metadata(&self, _pn: usize) -> PointCloudResult<Option<&Self::Metadata>> {
        Ok(None)
    }
    fn metasummary(&self, pns: &[usize]) -> PointCloudResult<SummaryCounter<Self::MetaSummary>> {
        Ok(SummaryCounter {
            summary: (),
            nones: pns.len(),
            errors: 0,
        })
    }
    fn label(&self, _pn: usize) -> PointCloudResult<Option<&Self::Label>> {
        Ok(None)
    }
    fn label_summary(&self, pns: &[usize]) -> PointCloudResult<SummaryCounter<Self::LabelSummary>> {
        Ok(SummaryCounter {
            summary: (),
            nones: pns.len(),
            errors: 0,
        })
    }
    fn name(&self, pi: usize) -> PointCloudResult<String> {
        Ok(pi.to_string())
    }
    fn index(&self, pn: &str) -> PointCloudResult<usize> {
        pn.parse::<usize>().map_err(|_| {
            ParsingError::RegularParsingError("Unable to parse your str into an usize").into()
        })
    }
    fn names(&self) -> Vec<String> {
        (0..self.len()).map(|i| i.to_string()).collect()
    }

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
    fn point<'a, 'b: 'a>(&'b self, pn: usize) -> PointCloudResult<Self::PointRef<'a>> {
        let lower_bound = self.row_index[pn].try_into();
        let upper_bound = self.row_index[pn + 1].try_into();
        if let (Ok(lower_bound), Ok(upper_bound)) = (lower_bound, upper_bound) {
            let values = &self.values[lower_bound..upper_bound];
            let indexes = &self.col_index[lower_bound..upper_bound];
            Ok(SparseRef::new(self.dim, values, indexes))
        } else {
            panic!("Could not covert a usize into a sparse dimension");
        }
    }
}
