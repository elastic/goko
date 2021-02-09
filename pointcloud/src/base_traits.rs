use std::sync::Mutex;

use rayon::prelude::*;
use std::cmp::min;
use std::fmt::Debug;

use std::ops::Deref;

use crate::pc_errors::*;
use serde::{Deserialize, Serialize};

/// A trait to ensure that we can create matrices and statiscial vectors from your point reference.
/// 
/// See [`crate::points`] for some pre-baked implementations. 
pub trait PointRef: Send + Sync {
    /// The iterator type for this reference.
    type DenseIter: Iterator<Item = f32>;

    /// provided because this could be faster than iteration (for example a memcpy).
    fn dense(&self) -> Vec<f32> {
        self.dense_iter().collect()
    }

    /// The actual call to the dense iterator that [`PointCloud`] uses.
    fn dense_iter(&self) -> Self::DenseIter;
}

/// Metric trait. Done as a trait so that it's easy to switch out.
/// 
/// Use a specific T. Don't implement it for a generic [S], but for [f32] or [u8], as you can use SIMD.
/// This library uses `packed_simd`. 
pub trait Metric<T: ?Sized>: Send + Sync + 'static {
    /// Distance calculator. Optimize the hell out of this if you're implementing it.
    fn dist(x: &T, y: &T) -> f32;
    // Implemented, but the system that uses this isn't yet.
    //fn norm(x: &RawSparse<f32, u32>) -> f32
}

use ndarray::{Array1, Array2};

#[inline]
fn chunk(data_dim: usize) -> usize {
    min(300000 / data_dim, 100)
}

/// Base trait for a point cloud
pub trait PointCloud: Send + Sync + 'static {
    /// The derefrenced, raw point. Think [f32]
    type Point: ?Sized;
    /// A reference to a point. Think &'a [f32]
    ///
    /// It might be better to move this to the `point` function, but this way we can rely on it for other things.
    type PointRef<'a>: Deref<Target = Self::Point> + PointRef;
    /// The metric this pointcloud is bound to. Think L2
    type Metric: Metric<Self::Point>;
    /// Name type, could be a string or a
    type Name: Sized + Clone + Eq;
    /// The label type.
    /// Summary of a set of labels
    type Label: ?Sized;
    /// Summary of a set of labels
    type LabelSummary: Summary<Label = Self::Label>;
    /// Underlying metadata
    type Metadata: ?Sized;
    /// A summary of the underlying metadata
    type MetaSummary: Summary<Label = Self::Metadata>;


    /// Expensive metadata object for the sample
    fn metadata(&self, pn: usize) -> PointCloudResult<Option<&Self::Metadata>>;
    /// Expensive metadata summary over the samples
    fn metasummary(&self, pns: &[usize]) -> PointCloudResult<SummaryCounter<Self::MetaSummary>>;

    /// Grabs a label reference. Supports errors (the label could be remote),
    /// and partially labeled datasets with the option.
    fn label(&self, pn: usize) -> PointCloudResult<Option<&Self::Label>>;
    /// Grabs a label summary of a set of indexes.
    fn label_summary(&self, pns: &[usize]) -> PointCloudResult<SummaryCounter<Self::LabelSummary>>;
    /// Grabs the name of the point.
    /// Returns an error if the access errors out, and a None if the name is unknown
    fn name(&self, pi: usize) -> PointCloudResult<Self::Name>;
    /// Converts a name to an index you can use
    fn index(&self, pn: &Self::Name) -> PointCloudResult<usize>;
    /// Gather's all valid known names
    fn names(&self) -> Vec<Self::Name>;

    /// The number of samples this cloud covers
    fn len(&self) -> usize;
    /// If this is empty
    fn is_empty(&self) -> bool;
    /// The dimension of the underlying data
    fn dim(&self) -> usize;
    /// Indexes used for access
    fn reference_indexes(&self) -> Vec<usize>;
    /// Gets a point from this dataset
    fn point<'a, 'b: 'a>(&'b self, i: usize) -> PointCloudResult<Self::PointRef<'a>>;

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
    ) -> PointCloudResult<Vec<f32>> {
        let chunk = chunk(self.dim());
        let mut dists: Vec<f32> = vec![f32::default(); is.len() * js.len()];
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
    fn distances_to_point_index(&self, i: usize, indexes: &[usize]) -> PointCloudResult<Vec<f32>> {
        self.distances_to_point(&self.point(i)?, indexes)
    }

    /// The main distance function. This paralizes if there are more than 100 points.
    fn distances_to_point<T: Deref<Target = Self::Point> + Send + Sync>(
        &self,
        x: &T,
        indexes: &[usize],
    ) -> PointCloudResult<Vec<f32>> {
        let chunk = chunk(self.dim());
        let len = indexes.len();
        if len > chunk * 3 {
            let mut dists: Vec<f32> = vec![f32::default(); indexes.len()];
            let dist_iter = dists.par_chunks_mut(chunk);
            let indexes_iter = indexes.par_chunks(chunk);
            let error: Mutex<Result<(), PointCloudError>> = Mutex::new(Ok(()));
            dist_iter
                .zip(indexes_iter)
                .for_each(|(chunk_dists, chunk_indexes)| {
                    for (d, i) in chunk_dists.iter_mut().zip(chunk_indexes) {
                        match self.point(*i).map(|y| Self::Metric::dist(&x, &y)) {
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
                    Ok(Self::Metric::dist(&x, &y))
                })
                .collect()
        }
    }

    /// The first moment of the specified vectors. See [wikipedia](https://en.wikipedia.org/wiki/Moment_(mathematics)).
    fn moment_1(&self, indexes: &[usize]) -> PointCloudResult<Vec<f32>>
    where
        f32: std::ops::AddAssign,
    {
        let mut moment_vec: Vec<f32> = vec![f32::default(); self.dim()];
        for i in indexes {
            match self.point(*i) {
                Ok(y) => {
                    for (m, yy) in moment_vec.iter_mut().zip(y.dense_iter()) {
                        *m += yy;
                    }
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }
        Ok(moment_vec)
    }

    /// The second moment of the specified vectors. See [wikipedia](https://en.wikipedia.org/wiki/Moment_(mathematics)).
    fn moment_2(&self, indexes: &[usize]) -> PointCloudResult<Vec<f32>>
    where
        f32: std::ops::Mul<Output = f32> + std::ops::AddAssign,
    {
        let mut moment_vec: Vec<f32> = vec![f32::default(); self.dim()];
        for i in indexes {
            match self.point(*i) {
                Ok(y) => {
                    for (m, yy) in moment_vec.iter_mut().zip(y.dense_iter()) {
                        *m += yy * yy;
                    }
                }
                Err(e) => {
                    return Err(e);
                }
            }
        }
        Ok(moment_vec)
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

impl Summary for () {
    type Label = ();
    fn add(&mut self, _v: &Self::Label) {}
    fn combine(&mut self, _other: &Self) {}
    fn count(&self) -> usize { 0 }
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
    fn label_summary(&self, pns: &[usize]) -> PointCloudResult<SummaryCounter<Self::LabelSummary>>;
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

impl<D, L> SimpleLabeledCloud<D, L> {
    /// Creates a new one
    pub fn new(data: D, labels: L) -> Self {
        SimpleLabeledCloud { data, labels }
    }
}

impl<D: PointCloud, L: LabelSet> PointCloud for SimpleLabeledCloud<D, L> {
    /// Underlying metric this point cloud uses
    type Metric = D::Metric;
    type Point = D::Point;
    type PointRef<'a> = D::PointRef<'a>;
    type Name = D::Name;
    type Metadata = D::Metadata;
    type MetaSummary = D::MetaSummary;

    type Label = L::Label;
    type LabelSummary = L::LabelSummary;

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
    fn point<'a, 'b: 'a>(&'b self, i: usize) -> PointCloudResult<Self::PointRef<'a>> {
        self.data.point(i)
    }

    fn metadata(&self, pn: usize) -> PointCloudResult<Option<&Self::Metadata>> {
        self.data.metadata(pn)
    }
    /// Expensive metadata summary over the samples
    fn metasummary(&self, pns: &[usize]) -> PointCloudResult<SummaryCounter<Self::MetaSummary>> {
        self.data.metasummary(pns)
    }

    /// Grabs a label reference. Supports errors (the label could be remote),
    /// and partially labeled datasets with the option.
    fn label(&self, pn: usize) -> PointCloudResult<Option<&Self::Label>> {
        self.labels.label(pn)
    }
    /// Grabs a label summary of a set of indexes.
    fn label_summary(&self, pns: &[usize]) -> PointCloudResult<SummaryCounter<Self::LabelSummary>> {
        self.labels.label_summary(pns)
    }
    /// Grabs the name of the point.
    /// Returns an error if the access errors out, and a None if the name is unknown
    fn name(&self, pi: usize) -> PointCloudResult<Self::Name> {
        self.data.name(pi)
    }
    /// Converts a name to an index you can use
    fn index(&self, pn: &Self::Name) -> PointCloudResult<usize> {
        self.data.index(pn)
    }
    /// Gather's all valid known names
    fn names(&self) -> Vec<Self::Name> {
        self.data.names()
    }
}

/// Enables the points in the underlying cloud to be named with strings.
pub trait NamedSet: Send + Sync + 'static {
    /// Name type, could be a string or a
    type Name: Sized + Clone + Eq;
    /// Number of elements in this name set
    fn len(&self) -> usize;
    /// If there are no elements left in this name set
    fn is_empty(&self) -> bool;
    /// Grabs the name of the point.
    /// Returns an error if the access errors out, and a None if the name is unknown
    fn name(&self, pi: usize) -> PointCloudResult<Self::Name>;
    /// Converts a name to an index you can use
    fn index(&self, pn: &Self::Name) -> PointCloudResult<usize>;
    /// Gather's all valid known names
    fn names(&self) -> Vec<Self::Name>;
}

/// Simply shoves together a point cloud and a name set, for a modular name system
#[derive(Debug)]
pub struct SimpleNamedCloud<D, N> {
    data: D,
    names: N,
}

impl<D: PointCloud, N: NamedSet> SimpleNamedCloud<D, N> {
    /// Creates a new one
    pub fn new(data: D, names: N) -> Self {
        assert_eq!(names.len(), data.len());
        SimpleNamedCloud { data, names }
    }
}

impl<D: PointCloud, N: NamedSet> PointCloud for SimpleNamedCloud<D, N> {
    /// Underlying metric this point cloud uses
    type Metric = D::Metric;
    type Point = D::Point;
    type PointRef<'a> = D::PointRef<'a>;
    type Name = N::Name;
    type Metadata = D::Metadata;
    type MetaSummary = D::MetaSummary;

    type Label = D::Label;
    type LabelSummary = D::LabelSummary;

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
    fn point<'a, 'b: 'a>(&'b self, i: usize) -> PointCloudResult<Self::PointRef<'a>> {
        self.data.point(i)
    }

    fn metadata(&self, pn: usize) -> PointCloudResult<Option<&Self::Metadata>> {
        self.data.metadata(pn)
    }
    /// Expensive metadata summary over the samples
    fn metasummary(&self, pns: &[usize]) -> PointCloudResult<SummaryCounter<Self::MetaSummary>> {
        self.data.metasummary(pns)
    }

    /// Grabs a label reference. Supports errors (the label could be remote),
    /// and partially labeled datasets with the option.
    fn label(&self, pn: usize) -> PointCloudResult<Option<&Self::Label>> {
        self.data.label(pn)
    }
    /// Grabs a label summary of a set of indexes.
    fn label_summary(&self, pns: &[usize]) -> PointCloudResult<SummaryCounter<Self::LabelSummary>> {
        self.data.label_summary(pns)
    }
    /// Grabs the name of the point.
    /// Returns an error if the access errors out, and a None if the name is unknown
    fn name(&self, pi: usize) -> PointCloudResult<Self::Name> {
        self.names.name(pi)
    }
    /// Converts a name to an index you can use
    fn index(&self, pn: &Self::Name) -> PointCloudResult<usize> {
        self.names.index(pn)
    }
    /// Gather's all valid known names
    fn names(&self) -> Vec<Self::Name> {
        self.names.names()
    }
}

/// Allows for expensive metadata, this is identical to the label trait, but enables slower update
pub trait MetaSet {
    /// Underlying metadata
    type Metadata: ?Sized;
    /// A summary of the underlying metadata
    type MetaSummary: Summary<Label = Self::Metadata>;

    /// Expensive metadata object for the sample
    fn metadata(&self, pn: usize) -> PointCloudResult<Option<&Self::Metadata>>;
    /// Expensive metadata summary over the samples
    fn metasummary(&self, pns: &[usize]) -> PointCloudResult<SummaryCounter<Self::MetaSummary>>;
}
