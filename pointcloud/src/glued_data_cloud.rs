use crate::errors::{PointCloudError,PointCloudResult};
use std::path::Path;

use crate::{PointRef, PointIndex, Metric};
use crate::distances::*;

use crate::base_traits::*;

use hashbrown::HashMap;
use fxhash::FxBuildHasher;


pub struct HashGluedCloud<D:PointCloud> {
    addresses: HashMap<PointIndex, (usize, PointIndex), FxBuildHasher>,
    data_sources: Vec<D>,
}

impl<D:PointCloud> HashGluedCloud<D> {
    pub fn new(data_sources: Vec<D>) -> HashGluedCloud<D> {
        let mut addresses = HashMap::with_hasher(FxBuildHasher::default());
        let mut pi: PointIndex = 0;
        for (i,source) in data_sources.iter().enumerate() {
            for j in 0..source.len() {
                addresses.insert(pi,(i,j as PointIndex));
            }
        }
        HashGluedCloud {
            addresses,
            data_sources
        }
    }

    pub fn take_data_sources(self) -> Vec<D> {
        self.data_sources
    }
}

impl<D:PointCloud> HashGluedCloud<D> {
    #[inline]
    fn get_address(&self, pn: PointIndex) -> PointCloudResult<(usize, PointIndex)> {
        match self.addresses.get(&pn) {
            Some((i, j)) => Ok((*i, *j)),
            None => Err(PointCloudError::DataAccessError {
                index: pn,
                reason: "address not found".to_string(),
            }),
        }
    }
}

impl<D:PointCloud> PointCloud for HashGluedCloud<D> {
    type Metric = D::Metric;
    /// Returns a slice corresponding to the point in question. Used for rarely referenced points,
    /// like outliers or leaves.
    fn point(&self, pn: PointIndex) -> PointCloudResult<PointRef> {
        let (i, j) = self.get_address(pn)?;
        self.data_sources[i].point(j)
    }

    /// Total number of points in the point cloud
    fn len(&self) -> usize {
        self.data_sources.iter().fold(0, |acc, mm| acc + mm.len())
    }

    /// Total number of points in the point cloud
    fn is_empty(&self) -> bool {
        if self.data_sources.is_empty() {
            false
        } else {
            self.data_sources.iter().all(|m| m.is_empty())
        }
    }

    /// The names of the data are currently a shallow wrapper around a usize.
    fn reference_indexes(&self) -> Vec<PointIndex> {
        self.addresses.keys().cloned().collect()
    }

    /// Dimension of the data in the point cloud
    fn dim(&self) -> usize {
        self.data_sources[0].dim()
    }
}

impl<D:LabeledCloud> LabeledCloud for HashGluedCloud<D> {
    type Label = D::Label;
    type LabelSummary = D::LabelSummary;

    fn label(&self, pn: PointIndex) -> PointCloudResult<Option<&Self::Label>> {
        let (i, j) = self.get_address(pn)?;
        self.data_sources[i].label(j)
    }
    fn label_summary(&self, pns: &[PointIndex]) -> PointCloudResult<Self::LabelSummary> {
        let mut summary = Self::LabelSummary::default();
        for pn in pns {
            let (i, j) = self.get_address(*pn)?;
            summary.add(self.data_sources[i].label(j));
        }
        Ok(summary)
    }
}