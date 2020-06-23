use std::sync::Mutex;

use rayon::prelude::*;
use std::cmp::min;
use std::fmt::Debug;

use crate::distances::*;
use crate::pc_errors::*;
use crate::*;

#[inline]
fn chunk(data_dim: usize) -> usize {
    min(15000 / data_dim, 20)
}

/// Base trait for a point cloud
pub trait PointCloud: Debug + Send + Sync + 'static {
    /// Underlying metric this point cloud uses
    type Metric: Metric;

    /// The number of samples this cloud covers
    fn len(&self) -> usize;
    /// If this is empty
    fn is_empty(&self) -> bool;
    /// The dimension of the underlying data
    fn dim(&self) -> usize;
    /// Indexes used for access
    fn reference_indexes(&self) -> Vec<PointIndex>;
    /// Gets a point from this dataset
    fn point(&self, pn: PointIndex) -> PointCloudResult<PointRef>;

    /// The main distance function. This paralizes if there are more than 100 points.
    fn distances_to_point_indices(
        &self,
        is: &[PointIndex],
        js: &[PointIndex],
    ) -> PointCloudResult<Vec<f32>> {
        let chunk = chunk(self.dim());
        let mut dists: Vec<f32> = vec![0.0; is.len() * js.len()];
        if is.len() * js.len() > chunk {
            let dist_iter = dists.par_chunks_mut(js.len());
            let indexes_iter = is.par_iter().map(|i| (i, js));
            let error: Mutex<Result<(), PointCloudError>> = Mutex::new(Ok(()));
            dist_iter
                .zip(indexes_iter)
                .for_each(|(chunk_dists, (i, chunk_indexes))| {
                    match self.point(*i) {
                        Ok(x) => {
                            for (d, j) in chunk_dists.iter_mut().zip(chunk_indexes) {
                                match self
                                    .point(*j)
                                    .map(|y| (Self::Metric::dist)(&x, &y))
                                    .flatten()
                                {
                                    Ok(dist) => *d = dist,
                                    Err(e) => {
                                        *error.lock().unwrap() = Err(e);
                                    }
                                }
                            }
                        }
                        Err(e) => {
                            *error.lock().unwrap() = Err(e);
                        }
                    };
                });
            (error.into_inner().unwrap())?;
        } else {
            for (k, i) in is.iter().enumerate() {
                let x = self.point(*i)?;
                for (l, j) in js.iter().enumerate() {
                    let y = self.point(*j)?;
                    dists[k * js.len() + l] = (Self::Metric::dist)(&x, &y)?;
                }
            }
        }
        Ok(dists)
    }

    /// The main distance function. This paralizes if there are more than 100 points.
    fn distances_to_point_index(
        &self,
        i: PointIndex,
        indexes: &[PointIndex],
    ) -> PointCloudResult<Vec<f32>> {
        self.distances_to_point(&self.point(i)?, indexes)
    }

    /// The main distance function. This paralizes if there are more than 100 points.
    fn distances_to_point<'a, T: Into<PointRef<'a>>>(
        &self,
        point: T,
        indexes: &[PointIndex],
    ) -> PointCloudResult<Vec<f32>> {
        let chunk = chunk(self.dim());
        let len = indexes.len();
        let x: PointRef<'a> = point.into();
        if len > chunk * 3 {
            let mut dists: Vec<f32> = vec![0.0; len];
            let dist_iter = dists.par_chunks_mut(chunk);
            let indexes_iter = indexes.par_chunks(chunk);
            let error: Mutex<Result<(), PointCloudError>> = Mutex::new(Ok(()));
            dist_iter
                .zip(indexes_iter)
                .for_each(|(chunk_dists, chunk_indexes)| {
                    for (d, i) in chunk_dists.iter_mut().zip(chunk_indexes) {
                        match self
                            .point(*i)
                            .map(|y| (Self::Metric::dist)(&x, &y))
                            .flatten()
                        {
                            Ok(dist) => *d = dist,
                            Err(e) => {
                                *error.lock().unwrap() = Err(e);
                            }
                        }
                    }
                });
            (error.into_inner().unwrap())?;
            Ok(dists)
        } else {
            indexes
                .iter()
                .map(|i| {
                    let y = self.point(*i)?;
                    (Self::Metric::dist)(&x, &y)
                })
                .collect()
        }
    }

    /// The main distance function. This paralizes if there are more than 100 points.
    fn moment_subset(&self, moment: i32, indexes: &[PointIndex]) -> PointCloudResult<Vec<f32>> {
        let mut moment_vec: Vec<f32> = vec![0.0; self.dim()];
        for i in indexes {
            match self.point(*i) {
                Ok(y) => match y {
                    PointRef::Dense(y_vals) => {
                        for (m, yy) in moment_vec.iter_mut().zip(y_vals) {
                            *m += yy.powi(moment);
                        }
                    }
                    PointRef::Sparse(y_vals, y_inds) => {
                        for (i, v) in y_inds.iter().zip(y_vals) {
                            moment_vec[*i as usize] += v.powi(moment);
                        }
                    }
                },
                Err(e) => {
                    return Err(e);
                }
            }
        }
        Ok(moment_vec)
    }

    /// The main distance function. This paralizes if there are more than 100 points.
    fn partial_adjacency_matrix(
        &self,
        is: &[PointIndex],
        js: &[PointIndex],
    ) -> PointCloudResult<AdjMatrix> {
        if !is.is_sorted() || !js.is_sorted() {
            return Err(PointCloudError::NotSorted);
        }

        let mut vals: Vec<f32> = Vec::new();
        let mut indexes: Vec<(PointIndex, PointIndex)> = Vec::new();
        for i in is.iter() {
            let x = self.point(*i)?;
            for j in js.iter() {
                if i < j {
                    if !indexes.contains(&(*i, *j)) {
                        let y = self.point(*j)?;
                        vals.push((Self::Metric::dist)(&x, &y)?);
                        indexes.push((*i, *j));
                    }
                } else if j < i {
                    if !indexes.contains(&(*j, *i)) {
                        let y = self.point(*j)?;
                        vals.push((Self::Metric::dist)(&x, &y)?);
                        indexes.push((*j, *i));
                    }
                }
            }
        }
        Ok(AdjMatrix { vals, indexes })
    }

    /// Returns a sparse adj matrix for the given points.
    fn adjacency_matrix(&self, mut indexes: &[PointIndex]) -> PointCloudResult<AdjMatrix> {
        if !indexes.is_sorted() {
            return Err(PointCloudError::NotSorted);
        }

        let capacity = indexes.len() * (indexes.len() - 1) / 2;
        let mut vals = Vec::with_capacity(capacity);
        let mut ret_indexes = Vec::with_capacity(capacity);
        while indexes.len() > 1 {
            let i = indexes[0];
            indexes = &indexes[1..];
            let distances = self.distances_to_point_index(i, &indexes)?;
            indexes.iter().zip(distances).for_each(|(j, d)| {
                ret_indexes.push((i, *j));
                vals.push(d);
            });
        }
        Ok(AdjMatrix {
            vals,
            indexes: ret_indexes,
        })
    }
}

/// A sparse adjacency matrix.
#[derive(Debug)]
pub struct AdjMatrix {
    /// The distances between the respective points, same order as indexes
    pub vals: Vec<f32>,
    /// The pairs of indexes for the distances in vals
    pub indexes: Vec<(PointIndex, PointIndex)>,
}

impl AdjMatrix {
    /// This gets by passing the smaller of the two indexes as the first element of
    /// the pair and the larger as the second.
    pub fn get(&self, i: PointIndex, j: PointIndex) -> Option<f32> {
        if i == j {
            Some(0.0)
        } else {
            let index = if i < j {
                self.indexes.binary_search(&(i, j))
            } else {
                self.indexes.binary_search(&(j, i))
            };
            match index {
                Ok(i) => Some(self.vals[i]),
                Err(_) => None,
            }
        }
    }

    /// Iterates over all distances and gets the minimum.
    pub fn min(&self) -> f32 {
        self.vals
            .iter()
            .fold(1.0 / 0.0, |a, v| if v < &a { *v } else { a })
    }
}

/// A summary for labels and metadata. You can make this an empty zero sized type for when you don't need it.
pub trait Summary<T: ?Sized>: Default {
    /// Adding a single value to the summary.
    fn add(&mut self, v: PointCloudResult<Option<&T>>);
    /// Merging several summaries of your data source together. This results in a summary of underlying column over
    /// the union of the indexes used to create the input summaries.
    fn combine(&mut self, other: Self);
    /// The number of elements this summary covers
    fn count(&self) -> usize;
    /// The number of elements that were unlabeled that this summary covers
    fn nones(&self) -> usize;
    /// The number of errors this summary covers that the label could not be found
    fn errors(&self) -> usize;
}

/// A trait for a container that just holds labels. Meant to be used in conjunction with `SimpleLabeledCloud` to be
/// and easy label or metadata object.
pub trait LabelSet: Debug + Send + Sync + 'static {
    /// Underlying type.
    type Label: ?Sized;
    /// Summary of a set of labels
    type LabelSummary: Summary<Self::Label>;

    /// Number of elements in this label set
    fn len(&self) -> usize;
    /// If there are no elements left in this label set
    fn is_empty(&self) -> bool;
    /// Grabs a label reference. Supports errors (the label could be remote),
    /// and partially labeled datasets with the option.
    fn label(&self, pn: PointIndex) -> PointCloudResult<Option<&Self::Label>>;
    /// Grabs a label summary of a set of indexes.
    fn label_summary(&self, pns: &[PointIndex]) -> PointCloudResult<Self::LabelSummary>;
}

/// A point cloud that is labeled
pub trait LabeledCloud: PointCloud {
    /// Underlying type.
    type Label: ?Sized;
    /// Summary of a set of labels
    type LabelSummary: Summary<Self::Label>;
    /// Grabs a label reference. Supports errors (the label could be remote),
    /// and partially labeled datasets with the option.
    fn label(&self, pn: PointIndex) -> PointCloudResult<Option<&Self::Label>>;
    /// Grabs a label summary of a set of indexes.
    fn label_summary(&self, pns: &[PointIndex]) -> PointCloudResult<Self::LabelSummary>;
}

/// Simply shoves together a point cloud and a label set, for a modular label system
#[derive(Debug)]
pub struct SimpleLabeledCloud<D: PointCloud, L: LabelSet> {
    data: D,
    labels: L,
}

impl<D: PointCloud, L: LabelSet> SimpleLabeledCloud<D, L> {
    /// Creates a new one
    pub fn new(data: D, labels: L) -> Self {
        SimpleLabeledCloud { data, labels }
    }
}

impl<D: PointCloud, L: LabelSet> PointCloud for SimpleLabeledCloud<D, L> {
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

impl<D: PointCloud, L: LabelSet> LabeledCloud for SimpleLabeledCloud<D, L> {
    type Label = L::Label;
    type LabelSummary = L::LabelSummary;

    fn label(&self, pn: PointIndex) -> PointCloudResult<Option<&Self::Label>> {
        self.labels.label(pn)
    }
    fn label_summary(&self, pns: &[PointIndex]) -> PointCloudResult<Self::LabelSummary> {
        self.labels.label_summary(pns)
    }
}

/*
pub trait NamedCloud: PointCloud {
    fn name(&self, pi: PointIndex) -> Option<&PointName>;
    fn index(&self, pn: PointName) -> Option<&PointIndex>;
    fn names(&self) -> Vec<PointName>;
}

pub trait MetaCloud: PointCloud {
    type Metadata: ?Sized;
    type MetaSummary: Summary<Label = Self::Metadata>;

    fn metadata(&self, pn: PointIndex) -> PointCloudResult<&Self::Metadata>;
    fn metasummary(&self, pns: &[PointIndex]) -> PointCloudResult<Self::MetaSummary>;
}
*/
