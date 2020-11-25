//! Simple gluing structs that abstracts away multi cloud access

use crate::pc_errors::{PointCloudError, PointCloudResult};

use crate::base_traits::*;

use fxhash::FxBuildHasher;
use hashbrown::HashMap;

/// For large numbers of underlying point clouds
#[derive(Debug)]
pub struct HashGluedCloud<D>
{
    addresses: HashMap<usize, (usize, usize), FxBuildHasher>,
    data_sources: Vec<D>,
}

impl<D: PointCloud> HashGluedCloud<D> {
    /// Creates a new one, preserves the order in the supplied vec.
    pub fn new(data_sources: Vec<D>) -> HashGluedCloud<D> {
        let mut addresses = HashMap::with_hasher(FxBuildHasher::default());
        let mut pi: usize = 0;
        for (i, source) in data_sources.iter().enumerate() {
            for j in 0..source.len() {
                addresses.insert(pi, (i, j as usize));
                pi += 1;
            }
        }
        HashGluedCloud {
            addresses,
            data_sources,
        }
    }
}

impl<D> HashGluedCloud<D> {
    /// Remaps the indexes, treats the first element of the pair as the old index, and the second as the new index
    pub fn reindex(&mut self, new_indexes:&[(usize,usize)]) -> PointCloudResult<()>{
        assert!(new_indexes.len() == self.addresses.len());
        let mut new_addresses = HashMap::with_hasher(FxBuildHasher::default());
        for (old_index,new_index) in new_indexes.iter() {
            match self.addresses.get(&old_index) {
                Some(addr) => {
                    new_addresses.insert(*new_index, *addr);
                },
                None => return Err(PointCloudError::DataAccessError {
                    index: *old_index,
                    reason: "address not found".to_string(),
                }),
            }
        }
        self.addresses = new_addresses;
        Ok(())
    }

    /// Borrows the underlying data sources
    pub fn data_sources(&self) -> &[D] {
        &self.data_sources
    }

    /// Extracts the underlying point clouds
    pub fn take_data_sources(self) -> Vec<D> {
        self.data_sources
    }

    #[inline]
    fn get_address(&self, pn: usize) -> PointCloudResult<(usize, usize)> {
        match self.addresses.get(&pn) {
            Some((i, j)) => Ok((*i, *j)),
            None => Err(PointCloudError::DataAccessError {
                index: pn,
                reason: "address not found".to_string(),
            }),
        }
    }
}

impl<Point, Field, D: PointCloud<Point, Field>> PointCloud<Point, Field> for HashGluedCloud<D> 
where
    Point: ?Sized + Send + Sync,
    Field: PartialOrd + Send + Sync + Default + Copy + Clone {
    type Metric = D::Metric;
    /// Returns a slice corresponding to the point in question. Used for rarely referenced points,
    /// like outliers or leaves.
    fn point(&self, pn: usize) -> PointCloudResult<&Point> {
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
    fn reference_indexes(&self) -> Vec<usize> {
        self.addresses.keys().cloned().collect()
    }

    /// Dimension of the data in the point cloud
    fn dim(&self) -> usize {
        self.data_sources[0].dim()
    }
}

impl<D: LabeledCloud> LabeledCloud for HashGluedCloud<D> {
    type Label = D::Label;
    type LabelSummary = D::LabelSummary;

    fn label(&self, pn: usize) -> PointCloudResult<Option<&Self::Label>> {
        let (i, j) = self.get_address(pn)?;
        self.data_sources[i].label(j)
    }
    fn label_summary(
        &self,
        pns: &[usize],
    ) -> PointCloudResult<SummaryCounter<Self::LabelSummary>> {
        let mut summary = SummaryCounter::<Self::LabelSummary>::default();
        for pn in pns {
            let (i, j) = self.get_address(*pn)?;
            summary.add(self.data_sources[i].label(j));
        }
        Ok(summary)
    }
}

impl<D: PointCloud + NamedCloud> NamedCloud for HashGluedCloud<D> {
    type Name = D::Name;

    fn name(&self, pi: usize) -> PointCloudResult<&Self::Name> {
        let (i, j) = self.get_address(pi)?;
        self.data_sources[i].name(j)
    }
    fn index(&self, pn: &Self::Name) -> PointCloudResult<&usize> {
        for data_source in &self.data_sources {
            let index = data_source.index(pn);
            if index.is_ok() {
                return index;
            }
        }
        Err(PointCloudError::UnknownName)
    }
    fn names(&self) -> Vec<Self::Name> where <D as NamedCloud>::Name: std::marker::Sized {
        self.addresses
            .values()
            .filter_map(|(i, j)| {
                self.data_sources[*i]
                    .name(*j)
                    .ok()
                    .map(|n| n.clone())
            })
            .collect()
    }
}

impl<D: PointCloud + MetaCloud> MetaCloud for HashGluedCloud<D> {
    type Metadata = D::Metadata;
    type MetaSummary = D::MetaSummary;

    fn metadata(&self, pn: usize) -> PointCloudResult<Option<&Self::Metadata>> {
        let (i, j) = self.get_address(pn)?;
        self.data_sources[i].metadata(j)
    }
    fn metasummary(
        &self,
        pns: &[usize],
    ) -> PointCloudResult<SummaryCounter<Self::MetaSummary>> {
        let mut summary = SummaryCounter::<Self::MetaSummary>::default();
        for pn in pns {
            let (i, j) = self.get_address(*pn)?;
            summary.add(self.data_sources[i].metadata(j));
        }
        Ok(summary)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::data_sources::tests::*;
    use crate::data_sources::*;
    use crate::label_sources::*;

    pub fn build_glue_random_labeled_test(
        partitions: usize,
        count: usize,
        data_dim: usize,
        labels_dim: usize,
    ) -> HashGluedCloud<SimpleLabeledCloud<DataRam, VecLabels>> {
        HashGluedCloud::new(
            (0..partitions)
                .map(|_i| build_ram_random_labeled_test(count, data_dim, labels_dim))
                .collect(),
        )
    }

    pub fn build_glue_random_test(
        partitions: usize,
        count: usize,
        data_dim: usize,
    ) -> HashGluedCloud<DataRam> {
        HashGluedCloud::new(
            (0..partitions)
                .map(|_i| build_ram_random_test(count, data_dim))
                .collect(),
        )
    }

    pub fn build_glue_fixed_labeled_test(
        partitions: usize,
        count: usize,
        data_dim: usize,
    ) -> HashGluedCloud<SimpleLabeledCloud<DataRam, SmallIntLabels>> {
        HashGluedCloud::new(
            (0..partitions)
                .map(|_i| build_ram_fixed_labeled_test(count, data_dim))
                .collect(),
        )
    }

    pub fn build_glue_fixed_test(
        partitions: usize,
        count: usize,
        data_dim: usize,
    ) -> HashGluedCloud<DataRam> {
        HashGluedCloud::new(
            (0..partitions)
                .map(|_i| build_ram_fixed_test(count, data_dim))
                .collect(),
        )
    }

    #[test]
    fn address_correct() {
        let pc = build_glue_fixed_test(5, 2, 3);
        println!("{:?}", pc);

        for i in &[1, 3, 5, 7, 9] {
            let address = pc.get_address(*i).unwrap();
            assert_eq!(address, ((i / 2) as usize, i % 2));
        }
    }

    #[test]
    fn point_correct() {
        let pc = build_glue_fixed_test(5, 2, 3);
        println!("{:?}", pc);

        for i in &[1, 3, 5, 7, 9] {
            let point = pc.point(*i).unwrap();
            for d in point {
                assert_approx_eq!(1.0, d);
            }
        }
    }

    #[test]
    fn label_correct() {
        let pc = build_glue_fixed_labeled_test(5, 2, 3);
        println!("{:?}", pc);

        for i in &[1, 3, 5, 7, 9] {
            let label = pc.label(*i).unwrap();
            assert_eq!(label, Some(&1));
        }
    }

    #[test]
    fn summary_correct() {
        let pc = build_glue_fixed_labeled_test(5, 2, 3);
        println!("{:?}", pc);

        let label_summary = pc.label_summary(&[1, 3, 5, 7, 9]).unwrap();
        println!("{:?}", label_summary);
        assert_eq!(label_summary.nones, 0);
        assert_eq!(label_summary.errors, 0);
        assert_eq!(label_summary.summary.items[0], (1, 5));
    }

    #[test]
    fn distance_correct() {
        let pc = build_glue_fixed_test(5, 2, 3);
        println!("{:?}", pc);

        let indexes = [1, 3, 5, 7, 9];
        let point = vec![0.0; 5];

        let dists = pc.distances_to_point(&point, &indexes).unwrap();
        for d in dists {
            assert_approx_eq!(3.0f32.sqrt(), d);
        }
        let dists = pc.distances_to_point_index(0, &indexes).unwrap();
        for d in dists {
            assert_approx_eq!(3.0f32.sqrt(), d);
        }
    }
}
