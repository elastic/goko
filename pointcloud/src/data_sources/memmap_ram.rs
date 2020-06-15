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
use crate::errors::{PointCloudError,PointCloudResult};
use std::fs::OpenOptions;
use std::path::{Path,PathBuf};
use std::marker::PhantomData;

use crate::{PointRef, PointIndex, Metric};
use crate::distances::*;

use crate::base_traits::*;
use crate::label_sources::VecLabels;


/// A thin wrapper to give a `Box<[f32]>` dimensionality.
#[derive(Debug)]
pub struct DataMemmap<M:Metric> {
    name: String,
    data: Mmapf32,
    dim: usize,
    metric: PhantomData<M>,
}

/// The data stored in ram. 
#[derive(Debug)]
pub struct DataRam<M:Metric> {
    name: String,
    data: Box<[f32]>,
    dim: usize,
    metric: PhantomData<M>,
}

impl<M:Metric> DataMemmap<M> {
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
        Ok(DataMemmap { name, data, dim, metric: PhantomData,})
    }


    pub fn convert_to_label(self) -> VecLabels {
        VecLabels::new(self.data.to_vec(),self.dim)
    }

    /// Reads and consumes this memmap and copies it into ram.
    pub fn convert_to_ram(self) -> DataRam<M> {
        let dim = self.dim;
        let name = self.name;
        let mut data = Vec::with_capacity(self.data.len());
        data.extend_from_slice(&*self.data);
        DataRam {
            name,
            data: Box::from(data),
            dim,
            metric: PhantomData,
        }
    }
}

impl<M:Metric> DataRam<M> {
    /// Consumes your box and dimension and gives a dimensioned box.
    pub fn new(dim: usize, data: Box<[f32]>) -> Result<DataRam<M>, PointCloudError> {
        assert!(data.len() % dim == 0);
        let name = "RAM".to_string();
        Ok(DataRam { name, data, dim, metric: PhantomData})
    }
}


macro_rules! make_point_cloud {
    ($name:ident) => {
        impl<M:Metric> PointCloud for $name<M> {
            type Metric = M;

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
            fn reference_indexes(&self) -> Vec<PointIndex> {
                (0..self.len()).map(|i| i as PointIndex).collect()
            }
            #[inline]
            fn point(&self, i: PointIndex) -> PointCloudResult<PointRef> {
                match self.data.get(self.dim * (i as usize)..(self.dim * (i as usize) + self.dim)) {
                    None => Err(PointCloudError::data_access(i as usize, self.name.clone())),
                    Some(x) => Ok(PointRef::Dense(x)),
                }
            }
        }
    };
}


make_point_cloud!(DataRam);
make_point_cloud!(DataMemmap);
