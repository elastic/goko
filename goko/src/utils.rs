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

//! Utility functions for i/o

use crate::errors::{GokoError, GokoResult};
use crate::tree_file_format::*;
use protobuf::{CodedInputStream, CodedOutputStream, Message};
use std::fs::File;
use std::fs::{read_to_string, remove_file, OpenOptions};
use std::path::Path;
use std::sync::Arc;
use yaml_rust::YamlLoader;

use crate::builders::CoverTreeBuilder;

use crate::CoverTreeWriter;

use pointcloud::loaders::{labeled_ram_from_yaml, ram_from_yaml};
use pointcloud::*;

/// Given a yaml file on disk, it builds a covertree.
///
/// ```yaml
/// ---
/// leaf_cutoff: 5
/// min_res_index: -10
/// scale_base: 1.3
/// data_path: DATAMEMMAPs
/// labels_path: LABELS_CSV
/// count: NUMBER_OF_DATA_POINTS
/// data_dim: 784
/// labels_index: 3
/// ```
pub fn cover_tree_from_labeled_yaml<P: AsRef<Path>>(
    path: P,
) -> GokoResult<CoverTreeWriter<DefaultLabeledCloud<L2>>> {
    let config = read_to_string(&path).expect("Unable to read config file");

    let params_files = YamlLoader::load_from_str(&config).unwrap();
    let params = &params_files[0];

    let point_cloud = labeled_ram_from_yaml::<_, L2>(&path)?;
    if let Some(count) = params["count"].as_i64() {
        if count as usize != point_cloud.len() {
            panic!(
                "We expected {:?} points, but the file has {:?} points at dim {:?}",
                count,
                point_cloud.len(),
                point_cloud.dim()
            );
        }
    }

    let builder = CoverTreeBuilder::from_yaml(&path);
    println!(
        "Loaded dataset, building a cover tree with scale base {}, leaf_cutoff {}, min_res_index {}, and use_singletons {}",
        &builder.scale_base, &builder.min_res_index, &builder.min_res_index, &builder.use_singletons
    );
    Ok(builder.build(Arc::new(point_cloud))?)
}

/// Given a yaml file on disk, it builds a covertree.
///
/// ```yaml
/// ---
/// leaf_cutoff: 5
/// min_res_index: -10
/// scale_base: 1.3
/// data_path: DATAMEMMAPs
/// count: NUMBER_OF_DATA_POINTS
/// data_dim: 784
/// ```

pub fn cover_tree_from_yaml<P: AsRef<Path>>(
    path: P,
) -> GokoResult<CoverTreeWriter<DefaultCloud<L2>>> {
    let config = read_to_string(&path).expect("Unable to read config file");

    let params_files = YamlLoader::load_from_str(&config).unwrap();
    let params = &params_files[0];

    let point_cloud = ram_from_yaml::<_, L2>(&path)?;
    if let Some(count) = params["count"].as_i64() {
        if count as usize != point_cloud.len() {
            panic!(
                "We expected {:?} points, but the file has {:?} points at dim {:?}",
                count,
                point_cloud.len(),
                point_cloud.dim()
            );
        }
    }
    let builder = CoverTreeBuilder::from_yaml(&path);
    println!(
        "Loaded dataset, building a cover tree with scale base {}, leaf_cutoff {}, min_res_index {}, and use_singletons {}",
        &builder.scale_base, &builder.min_res_index, &builder.min_res_index, &builder.use_singletons
    );
    Ok(builder.build(Arc::new(point_cloud))?)
}

/// Helper function that handles the file I/O and protobuf decoding for you.
pub fn load_tree<P: AsRef<Path>, D: PointCloud>(
    tree_path: P,
    point_cloud: Arc<D>,
) -> GokoResult<CoverTreeWriter<D>> {
    let tree_path_ref: &Path = tree_path.as_ref();
    println!("\nLoading tree from : {}", tree_path_ref.to_string_lossy());

    if !tree_path_ref.exists() {
        let tree_path_str = match tree_path_ref.to_str() {
            Some(expr) => expr,
            None => panic!("Unicode error with the tree path"),
        };
        panic!("{} does not exist\n", tree_path_str);
    }

    let mut cover_proto = CoreProto::new();

    let mut file = File::open(&tree_path_ref).map_err(GokoError::from)?;
    let mut cis = CodedInputStream::new(&mut file);
    if let Err(e) = cover_proto.merge_from(&mut cis) {
        panic!("Proto buff was unable to read {:#?}", e)
    }

    CoverTreeWriter::load(&cover_proto, point_cloud)
}

/// Helper function that handles the file I/O and protobuf encoding for you.
pub fn save_tree<P: AsRef<Path>, D: PointCloud>(
    tree_path: P,
    cover_tree: &CoverTreeWriter<D>,
) -> GokoResult<()> {
    let tree_path_ref: &Path = tree_path.as_ref();

    println!("Saving tree to : {}", tree_path_ref.to_string_lossy());
    if tree_path_ref.exists() {
        let tree_path_str = match tree_path_ref.to_str() {
            Some(expr) => expr,
            None => panic!("Unicode error with the tree path"),
        };
        println!("\t \t {:?} exists, removing", tree_path_str);
        remove_file(&tree_path).map_err(GokoError::from)?;
    }

    let cover_proto = cover_tree.save();

    let mut core_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&tree_path)
        .unwrap();

    let mut cos = CodedOutputStream::new(&mut core_file);
    cover_proto.write_to(&mut cos).map_err(GokoError::from)?;
    cos.flush().map_err(GokoError::from)?;
    Ok(())
}
