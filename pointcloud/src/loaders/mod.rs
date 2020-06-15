use std::path::{Path,PathBuf};


use crate::{PointRef, PointIndex, Metric};
use crate::distances::*;

use crate::base_traits::*;
use crate::label_sources::*;
use crate::glued_data_cloud::*;
use crate::data_sources::*;
use crate::errors::*;


pub fn open_labeled_memmaps<M:Metric>(
    data_dim: usize,
    label_dim: usize,
    data_paths: &[PathBuf],
    labels_paths: &[PathBuf],
    ram: bool,
) -> PointCloudResult<HashGluedCloud<SimpleLabeledCloud<DataMemmap<M>, VecLabels>>> {
    if data_paths.len() != labels_paths.len() {
        panic!(
            "Mismatch of label and data paths Data: {:?}, Labels: {:?}",
            data_paths, labels_paths
        );
    }
    let collection: PointCloudResult<Vec<SimpleLabeledCloud<DataMemmap<M>, VecLabels>>> = data_paths.iter().zip(labels_paths.iter()).map(|(dp,lp)| {
        let data = DataMemmap::<M>::new(data_dim, &dp)?;
        let labels = DataMemmap::<M>::new(label_dim, &lp)?.convert_to_label();
        Ok(SimpleLabeledCloud::new(data,labels))
    }).collect();
    Ok(HashGluedCloud::new(collection?))
}

pub fn open_memmaps<M:Metric>(
    data_dim: usize,
    label_dim: usize,
    data_paths: &[PathBuf],
    ram: bool,
) -> PointCloudResult<HashGluedCloud<DataMemmap<M>>> {
    let collection: PointCloudResult<Vec<DataMemmap<M>>> = data_paths.iter().map(|dp| {
        DataMemmap::<M>::new(data_dim, &dp)
    }).collect();
    Ok(HashGluedCloud::new(collection?))
}

pub fn convert_glued_memmap_to_ram<M:Metric>(glued_cloud:HashGluedCloud<DataMemmap<M>>) -> HashGluedCloud<DataRam<M>> {
    HashGluedCloud::new(glued_cloud.take_data_sources().drain(0..).map(|ds| ds.convert_to_ram()).collect())
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