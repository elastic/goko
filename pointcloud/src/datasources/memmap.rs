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
use super::DataSource;
use crate::errors::PointCloudError;
use std::fs::OpenOptions;
use std::path::Path;

/// This is a thin wrapper around `memmapf32` to give it dimensionality, and name so that if there are errors in this memmap we can notify the user.
#[derive(Debug)]
pub struct DataMemmap {
    name: String,
    data: Mmapf32,
    dim: usize,
}

impl DataMemmap {
    /// Creates a new one from a path. The name is the path.
    pub fn new(dim: usize, path: &Path) -> Result<DataMemmap, PointCloudError> {
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
        let data = unsafe { Mmapf32::map(&file).map_err(PointCloudError::from)}?;
        Ok(DataMemmap { name, data, dim })
    }

    /// Reads and consumes this memmap and copies it into ram.
    pub fn convert_to_ram(self) -> DataRam {
        let dim = self.dim;
        let name = self.name;
        let mut data = Vec::with_capacity(self.data.len());
        data.extend_from_slice(&*self.data);
        DataRam {
            name,
            data: Box::from(data),
            dim,
        }
    }
}

impl DataSource for DataMemmap {
    #[inline]
    fn get(&self, i: usize) -> Result<&[f32], PointCloudError> {
        match self.data.get(self.dim * i..(self.dim * i + self.dim)) {
            None => Err(PointCloudError::data_access(i, self.name.clone())),
            Some(x) => Ok(x),
        }
    }
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
    fn name(&self) -> String {
        self.name.clone()
    }
}

/// A thin wrapper to give a `Box<[f32]>` dimensionality.
#[derive(Debug)]
pub struct DataRam {
    name: String,
    data: Box<[f32]>,
    dim: usize,
}

impl DataRam {
    /// Consumes your box and dimension and gives a dimensioned box.
    pub fn new(dim: usize, data: Box<[f32]>) -> Result<DataRam, PointCloudError> {
        assert!(data.len() % dim == 0);
        let name = "RAM".to_string();
        Ok(DataRam { name, data, dim })
    }
}

impl DataSource for DataRam {
    #[inline]
    fn get(&self, i: usize) -> Result<&[f32], PointCloudError> {
        match self.data.get(self.dim * i..(self.dim * i + self.dim)) {
            None => Err(PointCloudError::data_access(i, self.name.clone())),
            Some(x) => Ok(x),
        }
    }
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
    fn name(&self) -> String {
        self.name.clone()
    }
}
