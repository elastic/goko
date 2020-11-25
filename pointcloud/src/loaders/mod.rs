//! Loaders for datasets. Just opens them up and returns a point cloud.

use std::path::{Path, PathBuf};

use crate::base_traits::*;
use crate::data_sources::*;
use crate::glued_data_cloud::*;
use crate::label_sources::*;
use crate::pc_errors::*;

mod yaml_loaders;
pub use yaml_loaders::*;
mod csv_loaders;
pub use csv_loaders::*;

/// Opens a set of memmaps of both data and labels
pub fn open_labeled_memmaps(
    data_dim: usize,
    label_dim: usize,
    data_paths: &[PathBuf],
    labels_paths: &[PathBuf],
) -> PointCloudResult<HashGluedCloud<SimpleLabeledCloud<DataMemmapL2, VecLabels>>> {
    if data_paths.len() != labels_paths.len() {
        panic!(
            "Mismatch of label and data paths Data: {:?}, Labels: {:?}",
            data_paths, labels_paths
        );
    }
    let collection: PointCloudResult<Vec<SimpleLabeledCloud<DataMemmapL2, VecLabels>>> =
        data_paths
            .iter()
            .zip(labels_paths.iter())
            .map(|(dp, lp)| {
                let data = DataMemmapL2::new(data_dim, &dp)?;
                let labels = DataMemmapL2::new(label_dim, &lp)?.convert_to_labels();
                Ok(SimpleLabeledCloud::new(data, labels))
            })
            .collect();
    Ok(HashGluedCloud::new(collection?))
}

/// Opens a set of memmaps of just data
pub fn open_memmaps(
    data_dim: usize,
    data_paths: &[PathBuf],
) -> PointCloudResult<HashGluedCloud<DataMemmapL2>> {
    let collection: PointCloudResult<Vec<DataMemmapL2>> = data_paths
        .iter()
        .map(|dp| DataMemmapL2::new(data_dim, &dp))
        .collect();
    Ok(HashGluedCloud::new(collection?))
}

/// Concatenates a glued data memmap to a single ram dataset
pub fn convert_glued_memmap_to_ram(
    glued_cloud: HashGluedCloud<DataMemmapL2>,
) -> DataRamL2 {
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