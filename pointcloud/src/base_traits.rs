use std::collections::HashMap;
use std::sync::Mutex;

use rayon::prelude::*;
use std::cmp::min;
use std::convert::AsRef;

use crate::distances::*;
use crate::errors::*;
use crate::*;


#[inline]
fn chunk(data_dim: usize) -> usize {
    min(15000 / data_dim, 20)
}

pub trait PointCloud: Send + Sync + 'static {
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
    fn distances_to_point(
        &self,
        x: &PointRef,
        indexes: &[PointIndex],
    ) -> PointCloudResult<Vec<f32>> {
        let chunk = chunk(self.dim());
        let len = indexes.len();
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
                            .map(|y| (Self::Metric::dist)(x, &y))
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
                    (Self::Metric::dist)(x, &y)
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

    fn adjacency_matrix(&self, mut indexes: &[PointIndex]) -> PointCloudResult<AdjMatrix> {
        if indexes.is_sorted() {
            let capacity = indexes.len()*(indexes.len()-1)/2;
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
            Ok(AdjMatrix { vals, indexes: ret_indexes })
        } else {
            Err(PointCloudError::NotSorted)
        }
    }
}

#[derive(Debug)]
pub struct AdjMatrix {
    pub vals: Vec<f32>,
    pub indexes: Vec<(PointIndex, PointIndex)>
}

impl AdjMatrix {
    /// This gets by passing the smaller of the two indexes as the first element of
    /// the pair and the larger as the second.
    pub fn get(&self, i: PointIndex, j: PointIndex) -> Option<f32> {
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

    /// Iterates over all distances and gets the minimum.
    pub fn min(&self) -> f32 {
        self.vals
            .iter()
            .fold(1.0 / 0.0, |a, v| if v < &a { *v } else { a })
    }
}

/// The actual trait the columnar data sources has to implement
pub trait Summary: Default {
    type Label: ?Sized;
    /// Adding a single value to the summary. When implementing please check that your value is compatible with your summary
    fn add(&mut self, v: PointCloudResult<Option<&Self::Label>>);
    /// Merging several summaries of your data source together. This results in a summary of underlying column over
    /// the union of the indexes used to create the input summaries.
    fn combine(&mut self, other: Self);
}

pub trait LabelSet: Send + Sync + 'static {
    type Label: ?Sized;
    type LabelSummary: Summary<Label = Self::Label>;

    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn label(&self, pn: PointIndex) -> PointCloudResult<Option<&Self::Label>>;
    fn label_summary(&self, pns: &[PointIndex]) -> PointCloudResult<Self::LabelSummary>;
}

pub trait LabeledCloud: PointCloud {
    type Label: ?Sized;
    type LabelSummary: Summary<Label = Self::Label>;
    fn label(&self, pn: PointIndex) -> PointCloudResult<Option<&Self::Label>>;
    fn label_summary(&self, pns: &[PointIndex]) -> PointCloudResult<Self::LabelSummary>;
}



pub trait NamedCloud: PointCloud {
    fn name(&self, pi: PointIndex) -> Option<&PointName>;
    fn index(&self, pn: PointName) -> Option<&PointIndex>;
    fn names(&self) -> Vec<PointName>;
}

pub trait MetaCloud: PointCloud {
    type Metadata;
    type MetaSummary;

    fn metadata(&self, pn: PointIndex) -> PointCloudResult<Self::Metadata>;
    fn metasummary(&self, pns: &[PointIndex]) -> PointCloudResult<Self::MetaSummary>;
}