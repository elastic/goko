//! Some label sets to modularly glue together with the data sources.

use crate::base_traits::*;
use crate::pc_errors::*;
use crate::summaries::*;
use crate::PointIndex;

/// Labels for a small number of categories, using ints
#[derive(Debug)]
pub struct SmallIntLabels {
    labels: Vec<u64>,
    mask: Option<Vec<bool>>,
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
        if let Some(mask) = &self.mask {
            if mask[pn] {
                Ok(self
                    .labels
                    .get(pn))
            } else {
                Ok(None)
            }
        } else {
            Ok(self
            .labels
            .get(pn))
        }
    }
    fn label_summary(&self, pns: &[PointIndex]) -> PointCloudResult<Self::LabelSummary> {
        let mut result = Self::LabelSummary::default();
        if let Some(mask) = &self.mask {
            for i in pns {
                if mask[*i] {
                    result.add(self.label(*i));
                } else {
                    result.add(Ok(None));
                }
            }
        } else {
            for i in pns {
                result.add(self.label(*i));
            }
        }
        Ok(result)
    }
}

impl SmallIntLabels {
    /// Creates a new vec label.
    pub fn new(labels: Vec<u64>, mask: Option<Vec<bool>>) -> SmallIntLabels {
        SmallIntLabels { labels, mask }
    }

    /// Merges 2 labels together
    pub fn merge(&mut self, other: &Self) {
        self.labels.extend(other.labels.iter());
        let mut replace_mask = false;
        match (self.mask.as_mut(),other.mask.as_ref()) {
            (Some(s_mask),Some(o_mask)) => s_mask.extend(o_mask),
            (Some(s_mask),None) => s_mask.extend(std::iter::repeat(false).take(other.labels.len())),
            (None,Some(_)) => replace_mask = true,
            (None,None) => {},
        }

        if replace_mask {
            let mut mask = std::iter::repeat(false).take(self.labels.len()).collect::<Vec<bool>>();
            mask.extend(other.mask.as_ref().unwrap().iter());
            self.mask = Some(mask)
        }
    }
}

/// Uses a vector to label your data. It can be 1 hot encoded, but if you do that you should use `SmallIntLabels`
#[derive(Debug)]
pub struct VecLabels {
    labels: Vec<f32>,
    mask: Option<Vec<bool>>,
    label_dim: usize,
}

impl VecLabels {
    /// Creates a new vec label.
    pub fn new(labels: Vec<f32>, label_dim: usize, mask: Option<Vec<bool>>) -> VecLabels {
        assert!(labels.len() % label_dim == 0);
        VecLabels { labels, label_dim, mask }
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
        if let Some(mask) = &self.mask {
            if mask[pn] {
                Ok(self
                    .labels
                    .get(self.label_dim * (pn as usize)..self.label_dim * (pn as usize + 1)))
            } else {
                Ok(None)
            }
        } else {
            Ok(self
            .labels
            .get(self.label_dim * (pn as usize)..self.label_dim * (pn as usize + 1)))
        }
        
    }
    fn label_summary(&self, pns: &[PointIndex]) -> PointCloudResult<Self::LabelSummary> {
        let mut result = Self::LabelSummary::default();
        if let Some(mask) = &self.mask {
            for i in pns {
                if mask[*i] {
                    result.add(self.label(*i));
                } else {
                    result.add(Ok(None));
                }
            }
        } else {
            for i in pns {
                result.add(self.label(*i));
            }
        }
        Ok(result)
    }
}
