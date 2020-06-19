//! Loaders for datasets. Just opens them up and returns a point cloud.

use std::path::{Path, PathBuf};

use crate::base_traits::*;
use crate::data_sources::*;
use crate::pc_errors::*;
use crate::glued_data_cloud::*;
use crate::label_sources::*;
use crate::Metric;

mod yaml_loaders;
pub use yaml_loaders::*;
mod csv_loaders;
pub use csv_loaders::*;

/// Opens a set of memmaps of both data and labels
pub fn open_labeled_memmaps<M: Metric>(
    data_dim: usize,
    label_dim: usize,
    data_paths: &[PathBuf],
    labels_paths: &[PathBuf],
) -> PointCloudResult<HashGluedCloud<SimpleLabeledCloud<DataMemmap<M>, VecLabels>>> {
    if data_paths.len() != labels_paths.len() {
        panic!(
            "Mismatch of label and data paths Data: {:?}, Labels: {:?}",
            data_paths, labels_paths
        );
    }
    let collection: PointCloudResult<Vec<SimpleLabeledCloud<DataMemmap<M>, VecLabels>>> =
        data_paths
            .iter()
            .zip(labels_paths.iter())
            .map(|(dp, lp)| {
                let data = DataMemmap::<M>::new(data_dim, &dp)?;
                let labels = DataMemmap::<M>::new(label_dim, &lp)?.convert_to_label();
                Ok(SimpleLabeledCloud::new(data, labels))
            })
            .collect();
    Ok(HashGluedCloud::new(collection?))
}

/// Opens a set of memmaps of just data
pub fn open_memmaps<M: Metric>(
    data_dim: usize,
    data_paths: &[PathBuf],
) -> PointCloudResult<HashGluedCloud<DataMemmap<M>>> {
    let collection: PointCloudResult<Vec<DataMemmap<M>>> = data_paths
        .iter()
        .map(|dp| DataMemmap::<M>::new(data_dim, &dp))
        .collect();
    Ok(HashGluedCloud::new(collection?))
}

/// Concatenates a glued data memmap to a single ram dataset
pub fn convert_glued_memmap_to_ram<M: Metric>(
    glued_cloud: HashGluedCloud<DataMemmap<M>>,
) -> DataRam<M> {
    glued_cloud
        .take_data_sources()
        .drain(0..)
        .map(|ds| ds.convert_to_ram())
        .fold_first(|mut a, b| {
            a.merge(b);
            a
        })
        .unwrap()
}

/*

impl<M: Metric> PointCloud<M> {
    /// Builds the point cloud from a collection of memmaps. If you want it to load all the data
    /// into ram, pass a true for the ram.
    pub fn from_memmap_files(
        data_dim: usize,
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
            let new_data: Box<dyn DataSource> = if ram {
                Box::new((DataMemmap::new(data_dim, &dp)?).convert_to_ram())
            } else {
                Box::new(DataMemmap::new(data_dim, &dp)?)
            };
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
            data_sources,
            label_sources,
            names_to_indexes,
            indexes_to_names,
            addresses,
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
            names_to_indexes,
            indexes_to_names,
            addresses,
            data_dim,
            loaded_centers: Mutex::new(IndexMap::new()),
            labels_scheme,
            chunk,
            metric: PhantomData,
        })
    }



    /// Runs `from_yaml` on the file at a given path
    pub fn from_file<P: AsRef<Path>>(path: P) -> PointCloudResult<PointCloud<M>> {
        let config = fs::read_to_string(&path)
            .unwrap_or_else(|_| panic!("Unable to read config file {:?}", &path.as_ref()));

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
}
*/
