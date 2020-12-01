/*
* Licensed to Elasticsearch B.V. under one or more contributor
* license agreements. See the NOTICE file distributed with
* this work for additional information regarding copyright
* ownership. Elasticsearch B.V. licenses this file to you under
* the Apache License, Version 2.0 (the "License"); you may
* not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*  http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/

use crate::errors::GokoResult;
use pointcloud::*;
use rand::seq::SliceRandom;
use rand::{thread_rng, Rng};
use std::cmp::Ordering;
use std::sync::Arc;

#[derive(Clone, Debug)]
pub(crate) enum CoveredData {
    FirstCoveredData(FirstCoveredData),
    NearestCoveredData(NearestCoveredData),
}

impl CoveredData {
    pub(crate) fn max_distance(&self) -> f32 {
        match &self {
            Self::FirstCoveredData(a) => a.max_distance(),
            Self::NearestCoveredData(a) => a.max_distance(),
        }
    }
    pub(crate) fn into_indexes(self) -> Vec<usize> {
        match self {
            Self::FirstCoveredData(a) => a.into_indexes(),
            Self::NearestCoveredData(a) => a.into_indexes(),
        }
    }

    pub(crate) fn len(&self) -> usize {
        match self {
            Self::FirstCoveredData(a) => a.len(),
            Self::NearestCoveredData(a) => a.len(),
        }
    }

    pub(crate) fn center_index(&self) -> usize {
        match &self {
            Self::FirstCoveredData(a) => a.center_index,
            Self::NearestCoveredData(a) => a.center_index,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct FirstCoveredData {
    dists: Vec<f32>,
    coverage: Vec<usize>,
    pub(crate) center_index: usize,
}

#[derive(Debug, Clone)]
pub(crate) struct UncoveredData {
    coverage: Vec<usize>,
}

impl UncoveredData {
    pub(crate) fn pick_center<D: PointCloud>(
        &mut self,
        radius: f32,
        point_cloud: &Arc<D>,
    ) -> GokoResult<FirstCoveredData> {
        let mut rng = thread_rng();
        let new_center: usize = rng.gen_range(0, self.coverage.len());
        let center_index = self.coverage.remove(new_center);
        let dists = point_cloud.distances_to_point_index(center_index, &self.coverage)?;

        let mut close_index = Vec::with_capacity(self.coverage.len());
        let mut close_dist = Vec::with_capacity(self.coverage.len());
        let mut far = Vec::new();
        for (i, d) in self.coverage.iter().zip(&dists) {
            if *d < radius {
                close_index.push(*i);
                close_dist.push(*d);
            } else {
                far.push(*i);
            }
        }
        let close = FirstCoveredData {
            coverage: close_index,
            dists: close_dist,
            center_index,
        };
        self.coverage = far;
        Ok(close)
    }

    pub(crate) fn len(&self) -> usize {
        self.coverage.len()
    }
}

fn find_split(dist_indexes: &[(f32, usize)], thresh: f32) -> usize {
    let mut smaller = 0;
    let mut larger = dist_indexes.len() - 1;

    while smaller <= larger {
        let m = (smaller + larger) / 2;
        if dist_indexes[m].0 < thresh {
            smaller = m + 1;
        } else if dist_indexes[m].0 > thresh {
            if m == 0 {
                return 0;
            }
            larger = m - 1;
        } else {
            return m;
        }
    }
    smaller
}

impl FirstCoveredData {
    pub(crate) fn new<D: PointCloud>(point_cloud: &Arc<D>) -> GokoResult<FirstCoveredData> {
        let mut coverage = point_cloud.reference_indexes();
        let center_index = coverage.pop().unwrap();
        let dists = point_cloud.distances_to_point_index(center_index, &coverage)?;
        Ok(FirstCoveredData {
            dists,
            coverage,
            center_index,
        })
    }

    pub(crate) fn split(self, thresh: f32) -> GokoResult<(FirstCoveredData, UncoveredData)> {
        let mut close_index = Vec::with_capacity(self.coverage.len());
        let mut close_dist = Vec::with_capacity(self.coverage.len());
        let mut far = Vec::new();
        for (i, d) in self.coverage.iter().zip(&self.dists) {
            if *d < thresh {
                close_index.push(*i);
                close_dist.push(*d);
            } else {
                far.push(*i);
            }
        }
        let close = FirstCoveredData {
            coverage: close_index,
            dists: close_dist,
            center_index: self.center_index,
        };
        let new_far = UncoveredData { coverage: far };
        Ok((close, new_far))
    }

    pub(crate) fn into_indexes(self) -> Vec<usize> {
        self.coverage
    }

    pub(crate) fn max_distance(&self) -> f32 {
        self.dists
            .iter()
            .cloned()
            .fold(-1. / 0. /* -inf */, f32::max)
    }

    pub(crate) fn len(&self) -> usize {
        self.coverage.len() + 1
    }
}

#[derive(Clone, Debug)]
pub(crate) struct NearestCoveredData {
    centers: Vec<usize>,
    dists: Vec<Vec<f32>>,
    point_indexes: Vec<usize>,
    center_dists: Vec<f32>,
    pub(crate) center_index: usize,
}

impl NearestCoveredData {
    pub(crate) fn new<D: PointCloud>(point_cloud: &Arc<D>) -> GokoResult<NearestCoveredData> {
        let mut point_indexes = point_cloud.reference_indexes();
        let center_index = point_indexes.pop().unwrap();
        let center_dists = point_cloud.distances_to_point_index(center_index, &point_indexes)?;
        let dists = vec![];
        let centers = vec![];
        Ok(NearestCoveredData {
            centers,
            dists,
            point_indexes,
            center_index,
            center_dists,
        })
    }

    fn cover_thyself<D: PointCloud>(
        &mut self,
        radius: f32,
        point_cloud: &Arc<D>,
    ) -> GokoResult<()> {
        let mut coverage: Vec<bool> = self.center_dists.iter().map(|d| d < &radius).collect();
        let mut rng = thread_rng();

        while coverage.iter().any(|b| !b) {
            let uncovered_indexes: Vec<usize> = self
                .point_indexes
                .iter()
                .zip(&coverage)
                .filter(|(_, b)| !**b)
                .map(|(pi, _)| *pi)
                .collect();
            let center_index = *uncovered_indexes.choose(&mut rng).unwrap();
            let new_dists =
                point_cloud.distances_to_point_index(center_index, &self.point_indexes)?;
            coverage
                .iter_mut()
                .zip(&new_dists)
                .for_each(|(a, d)| *a = *a || (d < &radius));
            self.dists.push(new_dists);
            self.centers.push(center_index);
        }

        Ok(())
    }

    fn add_point(&mut self, point_index: usize, distance: f32) {
        if point_index != self.center_index {
            self.center_dists.push(distance);
            self.point_indexes.push(point_index);
        }
    }

    fn assign_to_nearest(&self) -> (NearestCoveredData, Vec<NearestCoveredData>) {
        let mut new_center_coverage = NearestCoveredData {
            centers: vec![],
            dists: vec![],
            point_indexes: Vec::new(),
            center_index: self.center_index,
            center_dists: Vec::new(),
        };
        let mut new_coverage: Vec<NearestCoveredData> = self
            .centers
            .iter()
            .map(|center_index| NearestCoveredData {
                centers: vec![],
                dists: vec![],
                point_indexes: Vec::new(),
                center_index: *center_index,
                center_dists: Vec::new(),
            })
            .collect();

        for (i, pi) in self.point_indexes.iter().enumerate() {
            let (index, d) = self
                .dists
                .iter()
                .enumerate()
                .map(|(dist_index, dists)| (dist_index, dists[i]))
                .min_by(|(_di, d), (_ci, c)| d.partial_cmp(c).unwrap_or(Ordering::Equal))
                .unwrap_or((0, f32::MAX));
            if self.center_dists[i] < d {
                new_center_coverage.add_point(*pi, self.center_dists[i]);
            } else {
                new_coverage[index].add_point(*pi, d);
            }
        }

        (new_center_coverage, new_coverage)
    }

    pub(crate) fn split<D: PointCloud>(
        mut self,
        radius: f32,
        point_cloud: &Arc<D>,
    ) -> GokoResult<(NearestCoveredData, Vec<NearestCoveredData>)> {
        self.cover_thyself(radius, point_cloud)?;
        Ok(self.assign_to_nearest())
    }

    pub(crate) fn into_indexes(self) -> Vec<usize> {
        self.point_indexes
    }

    pub(crate) fn max_distance(&self) -> f32 {
        self.center_dists
            .iter()
            .cloned()
            .fold(-1. / 0. /* -inf */, f32::max)
    }

    pub(crate) fn len(&self) -> usize {
        self.point_indexes.len() + 1
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn splits_correctly_1() {
        let mut data = Vec::with_capacity(20);
        for _i in 0..19 {
            data.push(rand::random::<f32>() + 3.0);
        }
        data.push(0.0);

        let labels: Vec<i64> = data.iter().map(|x| if *x > 0.5 { 1 } else { 0 }).collect();

        let point_cloud = DefaultLabeledCloud::<L2>::new_simple(data, 1, labels);

        let cache = FirstCoveredData::new(&Arc::new(point_cloud)).unwrap();
        let (close, far) = cache.split(1.0).unwrap();

        assert_eq!(1, close.len());
        assert_eq!(19, far.len());
    }

    #[test]
    fn uncovered_splits_correctly_1() {
        let mut data = Vec::with_capacity(20);
        for _i in 0..19 {
            data.push(rand::random::<f32>() + 3.0);
        }
        data.push(0.0);

        let labels: Vec<i64> = data.iter().map(|x| if *x > 0.5 { 1 } else { 0 }).collect();

        let point_cloud = Arc::new(DefaultLabeledCloud::<L2>::new_simple(data, 1, labels));

        let mut cache = UncoveredData {
            coverage: (0..19 as usize).collect(),
        };
        let close = cache.pick_center(1.0, &point_cloud).unwrap();

        assert!(!close.coverage.contains(&close.center_index));
        assert!(!cache.coverage.contains(&close.center_index));
        for i in &close.coverage {
            assert!(!cache.coverage.contains(i));
        }
        for i in &cache.coverage {
            assert!(!close.coverage.contains(i));
        }
    }

    #[test]
    fn correct_dists() {
        let mut data = Vec::with_capacity(20);
        for _i in 0..19 {
            data.push(rand::random::<f32>() + 3.0);
        }
        data.push(0.0);

        let labels: Vec<i64> = data.iter().map(|x| if *x > 0.5 { 1 } else { 0 }).collect();

        //data.sort_unstable_by(|a, b| (a).partial_cmp(&b).unwrap_or(Ordering::Equal));
        let point_cloud = DefaultLabeledCloud::<L2>::new_simple(data.clone(), 1, labels);

        let cache = FirstCoveredData::new(&Arc::new(point_cloud)).unwrap();

        let thresh = 0.5;
        let mut true_close = Vec::new();
        let mut true_far = Vec::new();
        for i in 0..19 {
            if data[i] < thresh {
                true_close.push(i);
            } else {
                true_far.push(i);
            }
            assert_approx_eq!(data[i], cache.dists[i]);
        }
        let (close, _far) = cache.split(thresh).unwrap();

        for (tc, c) in true_close.iter().zip(close.coverage) {
            assert_eq!(*tc, c);
        }
    }

    #[test]
    fn nearest_splits_correctly_1() {
        let mut data = Vec::with_capacity(5);
        for _i in 0..4 {
            data.push(rand::random::<f32>() + 3.0);
        }
        data.push(0.0);

        let labels: Vec<i64> = data.iter().map(|x| if *x > 0.5 { 1 } else { 0 }).collect();

        let point_cloud = Arc::new(DefaultLabeledCloud::<L2>::new_simple(data, 1, labels));

        let mut cache = NearestCoveredData::new(&point_cloud).unwrap();
        cache.cover_thyself(1.0, &point_cloud).unwrap();

        assert_eq!(1, cache.dists.len());
        assert_eq!(4, cache.center_dists.len());
        assert_eq!(4, cache.dists[0].len());

        println!("{:#?}", cache);
        let (nested_split, splits) = cache.assign_to_nearest();
        println!("{:#?}", splits);

        assert_eq!(splits.len(), 1);
        assert_eq!(nested_split.len(), 1);
        assert_eq!(splits[0].len(), 4);
    }

    #[test]
    fn nearest_splits_nearest_1() {
        let cache = NearestCoveredData {
            center_index: 1,
            dists: vec![vec![0.0, 2.0, 0.0, 1.0, 2.0], vec![1.0, 0.0, 1.0, 2.0, 0.0]],
            point_indexes: vec![0, 2, 3, 4, 5],
            centers: vec![0, 2],
            center_dists: vec![2.0, 1.0, 2.0, 0.0, 1.0],
        };

        let (nested_split, splits) = cache.assign_to_nearest();

        println!("Nested Split: {:?}", nested_split);
        println!("Splits: {:?}", splits);
        assert_eq!(splits.len(), 2);
        assert_eq!(nested_split.center_index, 1);
        assert_eq!(splits[0].center_index, 0);
        assert_eq!(splits[1].center_index, 2);

        assert_eq!(nested_split.point_indexes[0], 4);
        assert_eq!(splits[0].point_indexes[0], 3);
        assert_eq!(splits[1].point_indexes[0], 5);

        assert_eq!(nested_split.center_dists, vec![0.0]);
        assert_eq!(splits[0].center_dists, vec![0.0]);
        assert_eq!(splits[1].center_dists, vec![0.0]);
    }

    /*
    #[test]
    fn correct_split_1() {
        for i in 0..100 {
            let mut dist_indexes:Vec<(f32,usize)> = Vec::with_capacity(20);
            for i in 0..2000 {
                dist_indexes.push((rand::random::<f32>(),i));
            }
            dist_indexes.sort_unstable_by(|a, b| (a.0).partial_cmp(&b.0).unwrap_or(Ordering::Equal));
            let thresh = 0.5;
            let split = find_split(&dist_indexes,thresh);
            let (close,far) = dist_indexes.split_at(split);
            for c in close {
                assert!(c.0 < thresh);
            }
            for f in far {
                assert!(f.0 > thresh);
            }
            assert!(close.len() + far.len() == dist_indexes.len());
        }
    }
    */
}
