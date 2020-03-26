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

//! The actual point cloud

use indexmap::IndexMap;
use std::collections::HashMap;
use std::fmt;
use std::fs::File;
use std::path::Path;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

use glob::{glob_with, MatchOptions};
use rayon::prelude::*;
use std::cmp::min;
use std::io::Read;
use std::marker::PhantomData;
use yaml_rust::{Yaml, YamlLoader};

use super::distances::*;
use super::errors::*;
use super::labels::values::*;
use super::labels::*;
use super::*;
use crate::datasources::DataSource;
use crate::datasources::*;
use crate::utils::*;

/// This abstracts away data access and the distance calculation. It handles both the labels and
/// points.
///
pub struct PointCloud<M: Metric> {
    addresses: IndexMap<PointIndex, (usize, usize)>,
    names_to_indexes: IndexMap<PointName, PointIndex>,
    indexes_to_names: IndexMap<PointIndex, PointName>,

    data_sources: Vec<Box<dyn DataSource>>,
    label_sources: Vec<MetadataList>,

    loaded_centers: Mutex<IndexMap<PointIndex, Arc<Vec<f32>>>>,
    data_dim: usize,
    labels_scheme: LabelScheme,
    chunk: usize,
    metric: PhantomData<M>,
}

impl<M: Metric> fmt::Debug for PointCloud<M> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "PointCloud {{ number of points: {}, number of memmaps: {}}}",
            self.addresses.len(),
            self.data_sources.len()
        )
    }
}

impl<M: Metric> PointCloud<M> {
    /// Builds the point cloud from a collection of memmaps. If you want it to load all the data
    /// into ram, pass a true for the ram.
    pub fn from_memmap_files(
        data_dim: usize,
        labels_scheme: LabelScheme,
        data_path: &[PathBuf],
        labels_path: &[PathBuf],
        ram: bool,
    ) -> PointCloudResult<PointCloud<M>> {
        if data_path.len() != labels_path.len() {
            panic!(
                "Mismatch of label and data paths Data: {:?}, Labels: {:?}",
                data_path, labels_path
            );
        }
        let mut addresses = IndexMap::new();
        let mut names_to_indexes: IndexMap<PointName, PointIndex> = IndexMap::new();
        let mut indexes_to_names: IndexMap<PointIndex, PointName> = IndexMap::new();
        let mut current_count: u64 = 0;
        let mut data_sources = Vec::new();
        let mut label_sources = Vec::new();
        for (i, (dp, lp)) in data_path.iter().zip(labels_path).enumerate() {
            let new_data: Box<dyn DataSource>;
            if ram {
                new_data = Box::new((DataMemmap::new(data_dim, &dp)?).convert_to_ram());
            } else {
                new_data = Box::new(DataMemmap::new(data_dim, &dp)?);
            }
            let new_labels = labels_scheme.open(&lp)?;
            if new_data.len() != new_labels.len() {
                panic!("The data count {:?} differs from the label count {:?} for the {}th data and label files", new_data.len(), new_labels.len(), i);
            }
            for j in 0..new_data.len() {
                let x = (i, j);
                let name = new_labels
                    .get_name(j)
                    .unwrap_or_else(|| format!("{}", current_count));

                if names_to_indexes.contains_key(&name) {
                    println!(
                        "Duplicate {:?} on line {} of file {:?}",
                        &name, j, labels_path[i]
                    );
                } else {
                    names_to_indexes.insert(name.clone(), current_count);
                    indexes_to_names.insert(current_count, name.clone());
                    addresses.insert(current_count, x);
                }
                current_count += 1;
            }
            data_sources.push(new_data);
            label_sources.push(new_labels);
        }

        // This could possibly be improved to be architecture specific. It depends on the CPU cache size
        let chunk = min(15000 / data_dim, 20);
        Ok(PointCloud {
            data_sources: data_sources,
            label_sources: label_sources,
            names_to_indexes: names_to_indexes,
            indexes_to_names: indexes_to_names,
            addresses: addresses,
            data_dim,
            labels_scheme,
            loaded_centers: Mutex::new(IndexMap::new()),
            chunk,
            metric: PhantomData,
        })
    }

    /// Builds the point cloud from data in ram.
    /// This is for complex metadata
    pub fn from_ram(
        data: Box<[f32]>,
        data_dim: usize,
        labels: MetadataList,
    ) -> PointCloudResult<PointCloud<M>> {
        let mut addresses = IndexMap::new();
        let data_source = Box::new(DataRam::new(data_dim, data)?);
        let labels_scheme = labels.scheme()?;
        let label_source = labels;
        let mut names_to_indexes: IndexMap<PointName, PointIndex> = IndexMap::new();
        let mut indexes_to_names: IndexMap<PointIndex, PointName> = IndexMap::new();

        for j in 0..(data_source.len()) {
            let name = label_source.get_name(j).unwrap_or_else(|| format!("{}", j));
            if names_to_indexes.contains_key(&name) {
                println!("Duplicate {:?} on line {} of file", &name, j);
            } else {
                names_to_indexes.insert(name.clone(), j as PointIndex);
                indexes_to_names.insert(j as PointIndex, name.clone());
                addresses.insert(j as u64, (0, j));
            }
        }
        let chunk = min(15000 / data_dim, 20);
        Ok(PointCloud {
            data_sources: vec![data_source],
            label_sources: vec![label_source],
            names_to_indexes: names_to_indexes,
            indexes_to_names: indexes_to_names,
            addresses: addresses,
            data_dim,
            loaded_centers: Mutex::new(IndexMap::new()),
            labels_scheme,
            chunk,
            metric: PhantomData,
        })
    }

    /// Given a yaml file on disk, it builds a point cloud. Minimal example below.
    /// ```yaml
    /// ---
    /// data_path: DATAMEMMAP
    /// labels_path: LABELS_CSV_OR_MEMMAP
    /// count: NUMBER_OF_DATA_POINTS
    /// data_dim: 784
    /// labels_dim: 10
    /// in_ram: True
    /// ```
    /// This assumes that your labels are either a CSV or a memmap file.
    /// If one specifies a schema then this is the minimal example
    /// ```yaml
    /// ---
    /// data_path: DATAMEMMAP
    /// labels_path: LABELS_CSV_OR_MEMMAP
    /// count: NUMBER_OF_DATA_POINTS
    /// data_dim: 784
    /// schema:
    ///    natural: u32
    ///    integer: i32
    ///    real: f32
    ///    string: String
    ///    boolean: bool
    /// ```
    pub fn from_yaml<P: AsRef<Path>>(
        params: &Yaml,
        yaml_path: P,
    ) -> PointCloudResult<PointCloud<M>> {
        let data_paths = &get_file_list(
            params["data_path"]
                .as_str()
                .expect("Unable to read the 'labels_path'"),
            yaml_path.as_ref(),
        );
        let labels_paths = &get_file_list(
            params["labels_path"]
                .as_str()
                .expect("Unable to read the 'labels_path'"),
            yaml_path.as_ref(),
        );
        let data_dim = params["data_dim"]
            .as_i64()
            .expect("Unable to read the 'data_dim'") as usize;

        let mut deser = LabelScheme::new();
        if params["schema"].is_badvalue() {
            let labels_dim = params["labels_dim"]
                .as_i64()
                .expect("Unable to read the 'labels_dim' or the 'schema'")
                as usize;
            deser.add_vector("y".to_string(), labels_dim, "f32");
        } else {
            build_label_schema_yaml(&mut deser, &params["schema"]);
        }

        let ram_bool = match params["in_ram"].as_bool() {
            Some(b) => b,
            None => true,
        };
        PointCloud::<M>::from_memmap_files(data_dim, deser, data_paths, labels_paths, ram_bool)
    }

    /// Runs `from_yaml` on the file at a given path
    pub fn from_file<P: AsRef<Path>>(path: P) -> PointCloudResult<PointCloud<M>> {
        let mut config_file =
            File::open(&path).expect(&format!("Unable to read config file {:?}", &path.as_ref()));
        let mut config = String::new();

        config_file
            .read_to_string(&mut config)
            .expect(&format!("Unable to read config file {:?}", &path.as_ref()));
        let params_files = YamlLoader::load_from_str(&config).unwrap();

        PointCloud::<M>::from_yaml(&params_files[0], path)
    }

    /// Builds the point cloud from data in ram. This is for quick things with simple metadata
    pub fn simple_from_ram(
        data: Box<[f32]>,
        data_dim: usize,
        labels: Box<[f32]>,
        labels_dim: usize,
    ) -> PointCloudResult<PointCloud<M>> {
        assert!(data.len() / data_dim == labels.len() / labels_dim);
        let list = MetadataList::simple_vec(labels, labels_dim);
        PointCloud::<M>::from_ram(data, data_dim, list)
    }

    /// Total number of points in the point cloud
    pub fn len(&self) -> usize {
        self.data_sources.iter().fold(0, |acc, mm| acc + mm.len())
    }

    /// Dimension of the data in the point cloud
    pub fn dim(&self) -> usize {
        self.data_dim
    }

    /// The names of the data are currently a shallow wrapper around a usize.
    pub fn reference_indexes(&self) -> Vec<PointIndex> {
        self.addresses.keys().cloned().collect()
    }

    /// Returns a arc that points to a AVX2 packed point. This also acts like a cache for these center
    /// points to ensure that we don't load multiple copies into memory. Used for heavily
    /// referenced points, like centers.
    pub fn get_center(&self, pn: PointIndex) -> PointCloudResult<Arc<Vec<f32>>> {
        let mut loaded_centers = self.loaded_centers.lock().unwrap();
        Ok(Arc::clone(
            loaded_centers
                .entry(pn)
                .or_insert(Arc::new(Vec::from(self.get_point(pn)?))),
        ))
    }

    #[inline]
    fn get_address(&self, pn: PointIndex) -> PointCloudResult<(usize, usize)> {
        match self.addresses.get(&pn) {
            Some((i, j)) => Ok((*i, *j)),
            None => panic!("Index not found"),
        }
    }

    /// Returns a slice corresponding to the point in question. Used for rarely referenced points,
    /// like outliers or leaves.
    pub fn get_point(&self, pn: PointIndex) -> PointCloudResult<&[f32]> {
        let (i, j) = self.get_address(pn)?;
        self.data_sources[i].get(j)
    }

    /// Gets the name from an index
    pub fn get_name(&self, pi: &PointIndex) -> Option<&PointName> {
        self.indexes_to_names.get(pi)
    }

    /// Gets the index from the name
    pub fn get_index(&self, pn: &PointName) -> Option<&PointIndex> {
        self.names_to_indexes.get(pn)
    }

    /// Gets all names in the point cloud
    pub fn get_names(&self) -> Vec<PointName> {
        self.names_to_indexes.keys().cloned().collect()
    }

    /// Gets a schema to use
    pub fn schema_json(&self) -> String {
        self.labels_scheme.schema_json()
    }

    /// Returns the label of a point.
    ///
    /// This will be changed to return a label structure that can contain many different pieces of info.
    pub fn get_metadata(&self, pn: PointIndex) -> PointCloudResult<Metadata> {
        let (i, j) = self.get_address(pn)?;
        self.label_sources[i].get(j)
    }

    /// Returns a complex summary of a collection of metadatas associated to a point
    pub fn get_metasummary(&self, pns: &[PointIndex]) -> PointCloudResult<MetaSummary> {
        let mut disk_splits: Vec<Vec<usize>> = vec![Vec::new(); self.label_sources.len()];
        for pn in pns.iter() {
            let (i, j) = self.get_address(*pn)?;
            disk_splits[i].push(j);
        }
        let disk_summaries: Vec<MetaSummary> = disk_splits
            .iter()
            .enumerate()
            .map(|(i, indexes)| self.label_sources[i].get_summary(indexes).unwrap())
            .collect();
        MetaSummary::combine(&disk_summaries)
    }

    /// The main distance function. This paralizes if there are more than 100 points.
    pub fn distances_to_point_indices(
        &self,
        is: &[PointIndex],
        js: &[PointIndex],
    ) -> PointCloudResult<Vec<f32>> {
        let mut dists: Vec<f32> = vec![0.0; is.len() * js.len()];
        if is.len() * js.len() > self.chunk {
            let dist_iter = dists.par_chunks_mut(js.len());
            let indexes_iter = is.par_iter().map(|i| (i, js));
            let error: Mutex<Result<(), PointCloudError>> = Mutex::new(Ok(()));
            dist_iter
                .zip(indexes_iter)
                .for_each(|(chunk_dists, (i, chunk_indexes))| {
                    match self.get_point(*i) {
                        Ok(x) => {
                            for (d, j) in chunk_dists.iter_mut().zip(chunk_indexes) {
                                match self.get_point(*j) {
                                    Ok(y) => *d = (M::dense)(x, y),
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
                let x = self.get_point(*i)?;
                for (l, j) in js.iter().enumerate() {
                    let y = self.get_point(*j)?;
                    dists[k * js.len() + l] = (M::dense)(x, y);
                }
            }
        }
        Ok(dists)
    }

    /// The main distance function. This paralizes if there are more than 100 points.
    pub fn distances_to_point_index(
        &self,
        i: PointIndex,
        indexes: &[PointIndex],
    ) -> PointCloudResult<Vec<f32>> {
        self.distances_to_point(self.get_point(i)?, indexes)
    }

    /// Create and adjacency matrix
    pub fn adj(&self, mut indexes: &[PointIndex]) -> PointCloudResult<AdjMatrix> {
        let mut vals = HashMap::new();
        while indexes.len() > 1 {
            let i = indexes[0];
            indexes = &indexes[1..];
            let distances = self.distances_to_point_index(i, &indexes)?;
            indexes.iter().zip(distances).for_each(|(j, d)| {
                if i < *j {
                    vals.insert((i, *j), d);
                } else {
                    vals.insert((*j, i), d);
                }
            });
        }
        Ok(AdjMatrix { vals })
    }

    /// The main distance function. This paralizes if there are more than 100 points.
    pub fn distances_to_point(
        &self,
        x: &[f32],
        indexes: &[PointIndex],
    ) -> PointCloudResult<Vec<f32>> {
        let len = indexes.len();
        if len > self.chunk * 3 {
            let mut dists: Vec<f32> = vec![0.0; len];
            let dist_iter = dists.par_chunks_mut(self.chunk);
            let indexes_iter = indexes.par_chunks(self.chunk);
            let error: Mutex<Result<(), PointCloudError>> = Mutex::new(Ok(()));
            dist_iter
                .zip(indexes_iter)
                .for_each(|(chunk_dists, chunk_indexes)| {
                    for (d, i) in chunk_dists.iter_mut().zip(chunk_indexes) {
                        match self.get_point(*i) {
                            Ok(y) => *d = (M::dense)(x, y),
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
                    let y = self.get_point(*i)?;
                    Ok((M::dense)(x, y))
                })
                .collect()
        }
    }

    /// The main distance function. This paralizes if there are more than 100 points.
    pub fn moment_subset(&self, moment: i32, indexes: &[PointIndex]) -> PointCloudResult<Vec<f32>> {
        let mut moment_vec: Vec<f32> = vec![0.0; self.data_dim];
        for i in indexes {
            match self.get_point(*i) {
                Ok(y) => {
                    for (m, yy) in moment_vec.iter_mut().zip(y) {
                        *m += yy.powi(moment);
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

fn build_label_schema_yaml(label_scheme: &mut LabelScheme, schema_yaml: &Yaml) {
    if let Some(schema_map) = schema_yaml.as_hash() {
        for (k, v) in schema_map.iter() {
            let key = k.as_str().unwrap().to_string();
            match v.as_str().unwrap() {
                "u32" => label_scheme.add_u32(key),
                "f32" => label_scheme.add_f32(key),
                "i32" => label_scheme.add_i32(key),
                "bool" => label_scheme.add_bool(key),
                "string" => label_scheme.add_string(key),
                "name" => label_scheme.add_name_column(&key),
                _ => panic!(
                    "Unknown type in schema yaml, also it should be (VALUE: TYPE): {:?}",
                    (k, v)
                ),
            }
        }
    } else {
        panic!("Need to correctly edit the yaml");
    }
}

fn get_file_list(files_reg: &str, yaml_path: &Path) -> Vec<PathBuf> {
    let options = MatchOptions {
        case_sensitive: false,
        ..Default::default()
    };
    let mut paths = Vec::new();
    let glob_paths;
    let files_reg_path = Path::new(files_reg);
    if files_reg_path.is_absolute() {
        glob_paths = match glob_with(&files_reg_path.to_str().unwrap(), options) {
            Ok(expr) => expr,
            Err(e) => panic!("Pattern reading error {:?}", e),
        };
    } else {
        glob_paths = match glob_with(
            &yaml_path
                .parent()
                .unwrap()
                .join(files_reg_path)
                .to_str()
                .unwrap(),
            options,
        ) {
            Ok(expr) => expr,
            Err(e) => panic!("Pattern reading error {:?}", e),
        };
    }

    for entry in glob_paths {
        let path = match entry {
            Ok(expr) => expr,
            Err(e) => panic!("Error reading path {:?}", e),
        };
        paths.push(path)
    }
    paths
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand;

    pub fn build_random_test(count: usize, data_dim: usize, labels_dim: usize) -> PointCloud<L2> {
        let data: Vec<f32> = (0..count * data_dim)
            .map(|_i| rand::random::<f32>())
            .collect();
        let labels: Vec<f32> = (0..count * labels_dim)
            .map(|_i| rand::random::<f32>())
            .collect();
        PointCloud::<L2>::simple_from_ram(Box::from(data), data_dim, Box::from(labels), labels_dim)
            .unwrap()
    }

    #[test]
    fn moment_correct() {
        let count = 10;
        let data_dim = 1;
        let labels_dim = 1;
        let data: Vec<f32> = (0..count * data_dim)
            .map(|_i| rand::random::<f32>())
            .collect();
        let mom1 = data.iter().fold(0.0, |a, x| a + x);
        let mom2 = data.iter().fold(0.0, |a, x| a + x * x);
        let labels: Vec<f32> = (0..count * labels_dim)
            .map(|_i| rand::random::<f32>())
            .collect();
        let pointcloud = PointCloud::<L2>::simple_from_ram(
            Box::from(data),
            data_dim,
            Box::from(labels),
            labels_dim,
        )
        .unwrap();

        let indexes = pointcloud.reference_indexes();
        let calc_mom1 = pointcloud.moment_subset(1, &indexes).unwrap();
        assert_eq!(mom1, calc_mom1[0]);
        let calc_mom2 = pointcloud.moment_subset(2, &indexes).unwrap();
        assert_eq!(mom2, calc_mom2[0]);
    }
}
