use std::sync::Mutex;

use rayon::prelude::*;
use std::cmp::min;
use std::fmt::Debug;

use std::ops::Deref;

use crate::pc_errors::*;
use serde::{Deserialize, Serialize};

pub trait PointRef<Field>: Send + Sync {
    type DenseIter: Iterator<Item = Field>;

    fn dense(&self) -> Vec<Field>;
    fn dense_iter(&self) -> Self::DenseIter;
}

pub trait Metric<T: ?Sized,S>: Send + Sync + 'static {
    fn dist(x: &T,y: &T) -> S;
    fn norm(x: &T) -> S;
}

//use ndarray::{Array1, Array2};

#[inline]
fn chunk(data_dim: usize) -> usize {
    min(300000 / data_dim, 100)
}

/// Base trait for a point cloud
pub trait PointCloud<T: ?Sized>: Send + Sync + 'static {
    /// Underlying metric this point cloud uses
    type Field: PartialOrd + Send + Sync + Default + Copy;
    type PointRef<'a>: Deref<Target = T> + PointRef<Self::Field>;
    type Metric: Metric<T,Self::Field>;

    /// The number of samples this cloud covers
    fn len(&self) -> usize;
    /// If this is empty
    fn is_empty(&self) -> bool;
    /// The dimension of the underlying data
    fn dim(&self) -> usize;
    /// Indexes used for access
    fn reference_indexes(&self) -> Vec<usize>;
    /// Gets a point from this dataset
    fn point<'a,'b:'a>(&'b self, i: usize) -> PointCloudResult<Self::PointRef<'a>>;

    /*
    /// Returns a dense array
    fn point_dense_array(&self, index: usize) -> PointCloudResult<Array1<f32>> {
        let pref = self.point(index)?;
        let vals: Vec<f32> = pref.dense_iter().collect();
        Ok(Array1::from_shape_vec((self.dim(),), vals).unwrap())
    }

    /// Returns a dense array
    fn points_dense_matrix(&self, indexes: &[usize]) -> PointCloudResult<Array2<f32>> {
        let dim = self.dim();
        let mut data: Vec<f32> = Vec::with_capacity(dim * indexes.len());
        for pi in indexes {
            let pref = self.point(*pi)?;
            data.extend(pref.dense_iter());
        }
        Ok(Array2::from_shape_vec((indexes.len(), dim), data).unwrap())
    }

    /// The main distance function. This paralizes if there are more than 100 points.
    fn moment_subset(&self, moment: i32, indexes: &[usize]) -> PointCloudResult<Vec<f32>> {
        let mut moment_vec: Vec<f32> = vec![f32::default(); self.dim()];
        for i in indexes {
            match self.point(*i) {
                Ok(y) => {
                    for (m, yy) in moment_vec.iter_mut().zip(y.dense_iter()) {
                        *m += yy.powi(moment);
                    }
                },
                Err(e) => {
                    return Err(e);
                }
            }
        }
        Ok(moment_vec)
    }
    */

    /*
    /// The main distance function. This paralizes if there are more than 100 points.
    fn partial_adjacency_matrix(
        &self,
        is: &[usize],
        js: &[usize],
    ) -> PointCloudResult<AdjMatrix> {
        if !is.is_sorted() || !js.is_sorted() {
            return Err(PointCloudError::NotSorted);
        }

        let mut vals: Vec<f32> = Vec::new();
        let mut indexes: Vec<(usize, usize)> = Vec::new();
        for i in is.iter() {
            let x = self.point(*i)?;
            for j in js.iter() {
                if i < j && !indexes.contains(&(*i, *j)) {
                    let y = self.point(*j)?;
                    vals.push(Self::Metric::dist(&x,&y));
                    indexes.push((*i, *j));
                } else if j < i && !indexes.contains(&(*j, *i)) {
                    let y = self.point(*j)?;
                    vals.push(Self::Metric::dist(&x,&y));
                    indexes.push((*j, *i));
                }
            }
        }
        Ok(AdjMatrix { vals, indexes })
    }

    /// Returns a sparse adj matrix for the given points.
    fn adjacency_matrix(&self, mut indexes: &[usize]) -> PointCloudResult<AdjMatrix> {
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
    */

    /*
    /// The main distance function. This paralizes if there are more than 100 points.
    fn distances_to_point_indices(
        &self,
        is: &[usize],
        js: &[usize],
    ) -> PointCloudResult<Vec<Self::Field>> {
        let chunk = chunk(self.dim());
        let mut dists: Vec<Self::Field> = vec![Self::Field::default(); is.len() * js.len()];
        if is.len() * js.len() > chunk {
            let dist_iter = dists.par_chunks_mut(js.len());
            let indexes_iter = is.par_iter().map(|i| (i, js));
            let error: Mutex<Result<(), PointCloudError>> = Mutex::new(Ok(()));
            dist_iter
                .zip(indexes_iter)
                .for_each(|(chunk_dists, (i, chunk_indexes))| {
                    match &self.point(*i) {
                        Ok(x) => {
                            for (d, j) in chunk_dists.iter_mut().zip(chunk_indexes) {
                                match self
                                    .point(*j)
                                    .map(|y| Self::Metric::dist(&x,&y))
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
                    dists[k * js.len() + l] = Self::Metric::dist(&x,&y);
                }
            }
        }
        Ok(dists)
    }
    */

    /// The main distance function. This paralizes if there are more than 100 points.
    fn distances_to_point_index(
        &self,
        i: usize,
        indexes: &[usize],
    ) -> PointCloudResult<Vec<Self::Field>> {
        self.distances_to_point(&(self.point(i)?), indexes)
    }

    /// The main distance function. This paralizes if there are more than 100 points.
    fn distances_to_point<'a,'b:'a>(
        &self,
        x: &Self::PointRef<'a>,
        indexes: &[usize],
    ) -> PointCloudResult<Vec<Self::Field>> {
        let chunk = chunk(self.dim());
        let len = indexes.len();
        if len > chunk * 3 {
            let mut dists: Vec<Self::Field> = vec![Self::Field::default(); indexes.len()];
            let dist_iter = dists.par_chunks_mut(chunk);
            let indexes_iter = indexes.par_chunks(chunk);
            let error: Mutex<Result<(), PointCloudError>> = Mutex::new(Ok(()));
            dist_iter
                .zip(indexes_iter)
                .for_each(|(chunk_dists, chunk_indexes)| {
                    for (d, i) in chunk_dists.iter_mut().zip(chunk_indexes) {
                        match self
                            .point(*i)
                            .map(|y| Self::Metric::dist(&x,&y))
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
                    Ok(Self::Metric::dist(&x,&y))
                })
                .collect()
        }
    }
}


/// A sparse adjacency matrix.
#[derive(Debug)]
pub struct AdjMatrix {
    /// The distances between the respective points, same order as indexes
    pub vals: Vec<f32>,
    /// The pairs of indexes for the distances in vals
    pub indexes: Vec<(usize, usize)>,
}

impl AdjMatrix {
    /// This gets by passing the smaller of the two indexes as the first element of
    /// the pair and the larger as the second.
    pub fn get(&self, i: usize, j: usize) -> Option<f32> {
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
pub trait Summary: Debug + Default + Send + Sync + 'static {
    /// Underlying type.
    type Label: ?Sized;
    /// Adding a single value to the summary.
    fn add(&mut self, v: &Self::Label);
    /// Merging several summaries of your data source together. This results in a summary of underlying column over
    /// the union of the indexes used to create the input summaries.
    fn combine(&mut self, other: &Self);
    /// The number of elements this summary covers
    fn count(&self) -> usize;
}

/// A trait for a container that just holds labels. Meant to be used in conjunction with `SimpleLabeledCloud` to be
/// and easy label or metadata object.
pub trait LabelSet: Debug + Send + Sync + 'static {
    /// Underlying type.
    type Label: ?Sized;
    /// Summary of a set of labels
    type LabelSummary: Summary<Label = Self::Label>;

    /// Number of elements in this label set
    fn len(&self) -> usize;
    /// If there are no elements left in this label set
    fn is_empty(&self) -> bool;
    /// Grabs a label reference. Supports errors (the label could be remote),
    /// and partially labeled datasets with the option.
    fn label(&self, pn: usize) -> PointCloudResult<Option<&Self::Label>>;
    /// Grabs a label summary of a set of indexes.
    fn label_summary(
        &self,
        pns: &[usize],
    ) -> PointCloudResult<SummaryCounter<Self::LabelSummary>>;
}

/// A point cloud that is labeled
pub trait LabeledCloud {
    /// Underlying type.
    type Label: ?Sized;
    /// Summary of a set of labels
    type LabelSummary: Summary<Label = Self::Label>;
    /// Grabs a label reference. Supports errors (the label could be remote),
    /// and partially labeled datasets with the option.
    fn label(&self, pn: usize) -> PointCloudResult<Option<&Self::Label>>;
    /// Grabs a label summary of a set of indexes.
    fn label_summary(
        &self,
        pns: &[usize],
    ) -> PointCloudResult<SummaryCounter<Self::LabelSummary>>;
}

/// Simply shoves together a point cloud and a label set, for a modular label system
#[derive(Default, Debug, Serialize, Deserialize)]
pub struct SummaryCounter<S: Summary> {
    /// The categorical summary
    pub summary: S,
    /// How many unlabeled elements this summary covers
    pub nones: usize,
    /// How many elements under this summary errored out
    pub errors: usize,
}

impl<S: Summary> SummaryCounter<S> {
    /// adds an element to the summary, handling errors
    pub fn add(&mut self, v: PointCloudResult<Option<&S::Label>>) {
        if let Ok(vv) = v {
            if let Some(val) = vv {
                self.summary.add(val);
            } else {
                self.nones += 1;
            }
        } else {
            self.errors += 1;
        }
    }

    /// Combines the underlying summaries, and the nones/errors
    pub fn combine(&mut self, other: &SummaryCounter<S>) {
        self.summary.combine(&other.summary);
        self.nones += other.nones;
        self.errors += other.errors;
    }

    /// a refernce to the underlying summary
    pub fn summary(&self) -> &S {
        &self.summary
    }

    /// the number of samples this summarieses
    pub fn count(&self) -> usize {
        self.summary.count() + self.nones + self.errors
    }

    /// how many unlabeled samples snuck thru
    pub fn nones(&self) -> usize {
        self.nones
    }

    /// how many samples labels errored out
    pub fn errors(&self) -> usize {
        self.errors
    }
}

/// Simply shoves together a point cloud and a label set, for a modular label system
#[derive(Debug)]
pub struct SimpleLabeledCloud<D, L> {
    data: D,
    labels: L,
}

impl<D, L: LabelSet> SimpleLabeledCloud<D, L> {
    /// Creates a new one
    pub fn new(data: D, labels: L) -> Self {
        SimpleLabeledCloud { data, labels }
    }
}

impl<T,D: PointCloud<T>, L: LabelSet> PointCloud<T> for SimpleLabeledCloud<D, L> {
    /// Underlying metric this point cloud uses
    type Field = D::Field;
    type Metric = D::Metric;
    type PointRef<'a> = D::PointRef<'a>;

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
    fn reference_indexes(&self) -> Vec<usize> {
        self.data.reference_indexes()
    }
    #[inline]
    fn point<'a,'b:'a>(&'b self, i: usize) -> PointCloudResult<Self::PointRef<'a>>{
        self.data.point(i)
    }
}

impl<D, L: LabelSet> LabeledCloud for SimpleLabeledCloud<D, L> {
    type Label = L::Label;
    type LabelSummary = L::LabelSummary;

    fn label(&self, pn: usize) -> PointCloudResult<Option<&Self::Label>> {
        self.labels.label(pn)
    }
    fn label_summary(
        &self,
        pns: &[usize],
    ) -> PointCloudResult<SummaryCounter<Self::LabelSummary>> {
        self.labels.label_summary(pns)
    }
}

/// Enables the points in the underlying cloud to be named with strings.
pub trait NamedCloud {
    /// Name type, could be a string or a 
    type Name: Sized + Clone +  Eq;
    /// Grabs the name of the point.
    /// Returns an error if the access errors out, and a None if the name is unknown
    fn name(&self, pi: usize) -> PointCloudResult<&Self::Name>;
    /// Converts a name to an index you can use
    fn index(&self, pn: &Self::Name) -> PointCloudResult<&usize>;
    /// Gather's all valid known names
    fn names(&self) -> Vec<Self::Name>;
}

/// Allows for expensive metadata, this is identical to the label trait, but enables slower update
pub trait MetaCloud {
    /// Underlying metadata
    type Metadata: ?Sized;
    /// A summary of the underlying metadata
    type MetaSummary: Summary<Label = Self::Metadata>;

    /// Expensive metadata object for the sample
    fn metadata(&self, pn: usize) -> PointCloudResult<Option<&Self::Metadata>>;
    /// Expensive metadata summary over the samples
    fn metasummary(
        &self,
        pns: &[usize],
    ) -> PointCloudResult<SummaryCounter<Self::MetaSummary>>;
}
