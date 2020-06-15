use crate::PointIndex;
use crate::summaries::*;
use crate::errors::*;
use crate::PointRef;
use crate::base_traits::*;



pub struct SmallIntLabels {
    labels: Vec<u64>
}

impl LabelSet for SmallIntLabels {
    type Label = u64;
    type LabelSummary = SmallCatSummary<u64>;

    fn len(&self) -> usize {
        self.labels.len()
    }
    fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }
    fn label(&self, pn: PointIndex) -> PointCloudResult<Option<&u64>> {
        Ok(self.labels.get(pn as usize))
    }
    fn label_summary(&self, pns: &[PointIndex]) -> PointCloudResult<Self::LabelSummary> {
        let mut result = Self::LabelSummary::default();
        for i in pns {
            result.add(Ok(self.labels.get(*i as usize)));
        }
        Ok(result)
    }
}

pub struct VecLabels {
    labels: Vec<f32>,
    label_dim: usize,
}

impl VecLabels {
    pub fn new(
        labels: Vec<f32>,
        label_dim: usize,
    ) -> VecLabels {
        VecLabels {
            labels,
            label_dim,
        }
    }
}

impl LabelSet for VecLabels {
    type Label = [f32];
    type LabelSummary = VecSummary;

    fn len(&self) -> usize {
        self.labels.len()
    }
    fn is_empty(&self) -> bool {
        self.labels.is_empty()
    }
    fn label(&self, pn: PointIndex) -> PointCloudResult<Option<&Self::Label>> {
        Ok(self.labels.get(self.label_dim*(pn as usize)..self.label_dim*(pn as usize+1)))
    }
    fn label_summary(&self, pns: &[PointIndex]) -> PointCloudResult<Self::LabelSummary> {
        let mut result = Self::LabelSummary::default();
        for i in pns {
            result.add(self.label(*i));
        }
        Ok(result)
    }
}

pub struct SimpleLabeledCloud<D: PointCloud, L: LabelSet> {
    data: D,
    labels: L,
}

impl<D: PointCloud, L: LabelSet> SimpleLabeledCloud<D,L> {
    pub fn new(data:D,labels:L) -> Self {
        SimpleLabeledCloud {
            data,
            labels,
        }
    }
}

impl<D:PointCloud,L:LabelSet> PointCloud for SimpleLabeledCloud<D,L> {
    type Metric = D::Metric;

    #[inline]
    fn dim(&self) -> usize {
        self.data.dim() 
    }
    #[inline]
    fn len(&self) -> usize {
        self.data.len()
    }
    #[inline]
    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
    #[inline]
    fn reference_indexes(&self) -> Vec<PointIndex> {
        self.data.reference_indexes()
    }
    #[inline]
    fn point(&self, i: PointIndex) -> PointCloudResult<PointRef> {
        self.data.point(i) 
    }
}

impl<D:PointCloud,L:LabelSet> LabeledCloud for SimpleLabeledCloud<D,L> {
    type Label = L::Label;
    type LabelSummary = L::LabelSummary;

    fn label(&self, pn: PointIndex) -> PointCloudResult<Option<&Self::Label>> {
        self.labels.label(pn)
    }
    fn label_summary(&self, pns: &[PointIndex]) -> PointCloudResult<Self::LabelSummary> {
        self.labels.label_summary(pns)
    }
}