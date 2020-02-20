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

use crate::errors::MalwareBrotResult;
use crate::tree_file_format::*;
use pointcloud::*;
use protobuf::{CodedInputStream, CodedOutputStream, Message};
use std::fs::File;
use std::fs::{remove_file, OpenOptions};
use std::io::Read;
use std::path::Path;
use yaml_rust::{Yaml, YamlLoader};

use crate::builders::CoverTreeBuilder;
use crate::errors::MalwareBrotError;
use crate::tree::CoverTreeWriter;

/// Given a yaml file on disk, it builds a covertree.
///
/// ```yaml
/// ---
/// cutoff: 5
/// resolution: -10
/// scale_base: 1.3
/// data_path: DATAMEMMAPs
/// labels_path: LABELS_CSVs
/// count: NUMBER_OF_DATA_POINTS
/// data_dim: 784
/// in_ram: True
/// file: mnist.tree (optional, if here and the file exists it loads it)
/// schema:
///    natural: u32
///    integer: i32
///    real: f32
///    string: String
///    boolean: bool
/// ```
/// Example without a schema:
/// ```yaml
/// ---
/// cutoff: 5
/// resolution: -10
/// scale_base: 1.3
/// data_path: DATAMEMMAPs
/// labels_path: LABELS_CSV_OR_MEMMAPs
/// count: NUMBER_OF_DATA_POINTS
/// data_dim: 784
/// labels_dim: 10
/// in_ram: True
/// file: mnist.tree (optional, if here and the file exists it loads it)
/// ```
///
pub fn cover_tree_from_yaml<P: AsRef<Path>>(path: P) -> MalwareBrotResult<CoverTreeWriter<L2>> {
    let mut config_file = File::open(&path).expect("Unable to open config file");

    let mut config = String::new();

    config_file
        .read_to_string(&mut config)
        .expect("Unable to read config file");
    let params_files = YamlLoader::load_from_str(&config).unwrap();
    let params = &params_files[0];

    let point_cloud: PointCloud<L2> = PointCloud::<L2>::from_yaml(&params)?;
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

    let (scale_base, cutoff, resolution, use_singletons) = read_ct_params_yaml(&params);
    println!(
        "Loaded dataset, building a cover tree with scale base {}, cutoff {}, min resolution {}, and use_singletons {}",
        scale_base, cutoff, resolution, use_singletons
    );
    let mut builder = CoverTreeBuilder::new();
    let verbosity = params["verbosity"].as_i64().unwrap_or(2) as u32;
    builder
        .set_scale_base(scale_base)
        .set_cutoff(cutoff)
        .set_resolution(resolution)
        .set_use_singletons(use_singletons)
        .set_verbosity(verbosity);
    Ok(builder.build(point_cloud)?)
}

/// Helper function for the above
pub fn read_ct_params_yaml(params: &Yaml) -> (f32, usize, i32, bool) {
    (
        params["scale_base"]
            .as_f64()
            .expect("Unable to read the 'scale_base' during yaml load") as f32,
        params["cutoff"]
            .as_i64()
            .expect("Unable to read the 'cutoff'") as usize,
        params["resolution"]
            .as_i64()
            .expect("Unable to read the 'resolution'") as i32,
        params["use_singletons"]
            .as_bool()
            .expect("Unable to read the 'use_singletons'"),
    )
}

/// Helper function that handles the file I/O and protobuf decoding for you.
pub fn load_tree<P: AsRef<Path>, M: Metric>(
    tree_path: P,
    point_cloud: PointCloud<M>,
) -> MalwareBrotResult<CoverTreeWriter<M>> {
    let tree_path_ref: &Path = tree_path.as_ref();
    println!("\nLoading tree from : {}", tree_path_ref.to_string_lossy());

    if !tree_path_ref.exists() {
        let tree_path_str = match tree_path_ref.to_str() {
            Some(expr) => expr,
            None => panic!("Unicode error with the tree path"),
        };
        panic!(tree_path_str.to_string() + &" does not exist\n".to_string());
    }

    let mut cover_proto = CoreProto::new();

    let mut file = match File::open(&tree_path_ref) {
        Ok(file) => file,
        Err(e) => panic!("Unable to open file {:#?}", e),
    };
    let mut cis = CodedInputStream::new(&mut file);
    if let Err(e) = cover_proto.merge_from(&mut cis) {
        panic!("Proto buff was unable to read {:#?}", e)
    }

    CoverTreeWriter::load(&cover_proto, point_cloud)
}

/// Helper function that handles the file I/O and protobuf encoding for you.
pub fn save_tree<P: AsRef<Path>, M: Metric>(
    tree_path: P,
    cover_tree: &CoverTreeWriter<M>,
) -> MalwareBrotResult<()> {
    let tree_path_ref: &Path = tree_path.as_ref();

    println!("Saving tree to : {}", tree_path_ref.to_string_lossy());
    if tree_path_ref.exists() {
        let tree_path_str = match tree_path_ref.to_str() {
            Some(expr) => expr,
            None => panic!("Unicode error with the tree path"),
        };
        println!("\t \t {:?} exists, removing", tree_path_str);
        remove_file(&tree_path).map_err(|e| MalwareBrotError::from(e))?;
    }

    let cover_proto = cover_tree.save();

    let mut core_file = match OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&tree_path)
    {
        Ok(core_file) => core_file,
        Err(_) => {
            panic!("unable to open {:?}", &tree_path_ref);
        }
    };

    let mut cos = CodedOutputStream::new(&mut core_file);
    cover_proto
        .write_to(&mut cos)
        .map_err(|e| MalwareBrotError::from(e))?;
    cos.flush().map_err(|e| MalwareBrotError::from(e))?;
    Ok(())
}
