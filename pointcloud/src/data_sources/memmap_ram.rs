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

//! Memmapped and Ram allocated data.

use super::memmapf32::Mmapf32;
use crate::pc_errors::{PointCloudError, PointCloudResult};
use std::fs::OpenOptions;
use std::marker::PhantomData;
use std::path::Path;

use crate::metrics::*;

use crate::base_traits::*;
use crate::label_sources::VecLabels;



/// A thin wrapper to give a `Box<[f32]>` dimensionality.
#[derive(Debug)]
pub struct DataMemmap<M = L2> {
    name: String,
    data: Mmapf32,
    dim: usize,
    metric: PhantomData<M>,
}

/// The data stored in ram.
#[derive(Debug)]
pub struct DataRam<M = L2> {
    name: String,
    data: Vec<f32>,
    dim: usize,
    metric: PhantomData<M>,
}

impl<M> DataMemmap<M> {
    /// Creates a new one from a path. The name is the path.
    pub fn new(dim: usize, path: &Path) -> PointCloudResult<DataMemmap<M>> {
        let name = path.to_string_lossy().to_string();
        if !path.exists() {
            panic!("data file {:?} does not exist", path);
        }
        let file = match OpenOptions::new().read(true).write(true).open(&path) {
            Ok(file) => file,
            Err(er) => {
                panic!("unable to open {:?} in from_proto, {:?}", path, er);
            }
        };
        let data = unsafe { Mmapf32::map(&file).map_err(PointCloudError::from) }?;
        Ok(DataMemmap {
            name,
            data,
            dim,
            metric: PhantomData,
        })
    }

    /// Reads and consumes this memmap and copies it into ram, then returns it to a labelset
    pub fn convert_to_labels(self) -> VecLabels {
        VecLabels::new(self.data.to_vec(), self.dim, None)
    }

    /// Reads and consumes this memmap and copies it into ram.
    pub fn convert_to_ram(self) -> DataRam {
        let dim = self.dim;
        let name = self.name;
        let mut data = Vec::with_capacity(self.data.len());
        data.extend_from_slice(&*self.data);
        DataRam {
            name,
            data,
            dim,
            metric: PhantomData,
        }
    }
}

impl<M> DataRam<M> {
    /// Consumes your box and dimension and gives a dimensioned box.
    pub fn new(data: Vec<f32>, dim: usize) -> Result<DataRam<M>, PointCloudError> {
        assert!(data.len() % dim == 0);
        let name = "RAM".to_string();
        Ok(DataRam {
            name,
            data,
            dim,
            metric: PhantomData,
        })
    }

    /// Converts this to a label set
    pub fn convert_to_labels(self) -> VecLabels {
        VecLabels::new(self.data, self.dim, None)
    }

    /// Merges two ram sets together.
    pub fn merge(&mut self, other: DataRam) {
        assert!(self.dim == other.dim);
        self.data.extend(other.data);
    }
}


macro_rules! make_point_cloud {
    ($name:ident) => {
        impl<M: Metric<[f32],f32>> PointCloud for $name<M> {
            type Metric = M;
            type Field = f32;
            type Point = [f32];
            type PointRef<'a> = &'a [f32];

            #[inline]
            fn dim(&self) -> usize {
                self.dim
            }
            #[inline]
            fn len(&self) -> usize {
                self.data.len() / self.dim
            }
            #[inline]
            fn is_empty(&self) -> bool {
                self.data.is_empty()
            }
            #[inline]
            fn reference_indexes(&self) -> Vec<usize> {
                (0..self.len()).map(|i| i as usize).collect()
            }
            #[inline]
            fn point<'a, 'b: 'a>(&'b self, i: usize) -> PointCloudResult<&'a [f32]> {
                match self.data
                    .get(self.dim * (i as usize)..(self.dim * (i as usize) + self.dim)) {
                    None => Err(PointCloudError::data_access(i as usize, self.name.clone())),
                    Some(x) => Ok(x),
                }
            }
        }
    };
}

make_point_cloud!(DataRam);
make_point_cloud!(DataMemmap);

#[cfg(test)]
pub mod tests {
    use super::*;
    use crate::label_sources::SmallIntLabels;
    use rand;
    use std::iter;

    pub fn build_ram_random_labeled_test(
        count: usize,
        data_dim: usize,
        labels_dim: usize,
    ) -> SimpleLabeledCloud<DataRam, VecLabels> {
        let data = DataRam::new(
            (0..count * data_dim)
                .map(|_i| rand::random::<f32>())
                .collect(),
            data_dim,
        )
        .unwrap();
        let labels = VecLabels::new(
            (0..count * labels_dim)
                .map(|_i| rand::random::<f32>())
                .collect(),
            labels_dim,
            None,
        );

        SimpleLabeledCloud::new(data, labels)
    }

    pub fn build_ram_random_test(count: usize, data_dim: usize) -> DataRam {
        DataRam::new(
            (0..count * data_dim)
                .map(|_i| rand::random::<f32>())
                .collect(),
            data_dim,
        )
        .unwrap()
    }

    pub fn build_ram_fixed_labeled_test(
        count: usize,
        data_dim: usize,
    ) -> SimpleLabeledCloud<DataRam, SmallIntLabels> {
        let data = DataRam::new(
            (0..count)
                .map(|i| iter::repeat(i as f32).take(data_dim))
                .flatten()
                .collect(),
            data_dim,
        )
        .unwrap();
        let labels = SmallIntLabels::new((0..count).map(|i| i as i64).collect(), None);

        SimpleLabeledCloud::new(data, labels)
    }

    pub fn build_ram_fixed_test(count: usize, data_dim: usize) -> DataRam {
        DataRam::new(
            (0..count)
                .map(|i| iter::repeat(i as f32).take(data_dim))
                .flatten()
                .collect(),
            data_dim,
        )
        .unwrap()
    }

    #[test]
    fn point_correct() {
        let pc = build_ram_fixed_test(5, 5);

        let point = pc.point(1).unwrap();
        for d in point.iter() {
            assert_approx_eq!(1.0, d);
        }
    }

    /*
    #[test]
    fn adjacency_correct() {
        let pc = build_ram_fixed_test(10, 5);

        let indexes: [usize; 5] = [1, 3, 5, 7, 9];

        let adj = pc.adjacency_matrix(&indexes).unwrap();
        println!("{:?}", adj);
        for val1 in &indexes {
            for val2 in &indexes {
                let diff = *val1 as f32 - *val2 as f32;
                let dist = (5.0 * (diff * diff).abs()).sqrt();
                let got_dist = pc.distances_to_point_index(*val1, &[*val2]).unwrap()[0];
                assert_approx_eq!(got_dist, adj.get(*val1, *val2).unwrap());
                assert_approx_eq!(dist, adj.get(*val1, *val2).unwrap());
            }
        }
    }
    */

    #[test]
    fn distance_correct() {
        let pc = build_ram_fixed_test(5, 5);

        let indexes = [1];
        let point = vec![0.0f32; 5];

        let dists = pc.distances_to_point(&&point[..], &indexes).unwrap();
        for d in dists {
            assert_approx_eq!(5.0f32.sqrt(), d);
        }
        let dists = pc.distances_to_point_index(0, &indexes).unwrap();
        for d in dists {
            assert_approx_eq!(5.0f32.sqrt(), d);
        }
    }
}
