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

use crate::errors::{GrandmaError, GrandmaResult};
use crate::tree_file_format::*;
use pointcloud::*;
use protobuf::{CodedInputStream, CodedOutputStream, Message};
use std::fs::File;
use std::fs::{read_to_string,remove_file, OpenOptions};
use std::path::Path;
use yaml_rust::{Yaml, YamlLoader};

use crate::builders::CoverTreeBuilder;

use crate::tree::CoverTreeWriter;

/// Given a yaml file on disk, it builds a covertree.
///
/// ```yaml
/// ---
/// leaf_cutoff: 5
/// min_res_index: -10
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
/// leaf_cutoff: 5
/// min_res_index: -10
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
pub fn cover_tree_from_yaml<P: AsRef<Path>>(path: P) -> GrandmaResult<CoverTreeWriter<L2>> {
    let config = read_to_string(&path).expect("Unable to read config file");
        
    let params_files = YamlLoader::load_from_str(&config).unwrap();
    let params = &params_files[0];

    let point_cloud: PointCloud<L2> = PointCloud::<L2>::from_yaml(&params, path)?;
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

    let (scale_base, leaf_cutoff, min_res_index, use_singletons) = read_ct_params_yaml(&params);
    println!(
        "Loaded dataset, building a cover tree with scale base {}, leaf_cutoff {}, min min_res_index {}, and use_singletons {}",
        scale_base, leaf_cutoff, min_res_index, use_singletons
    );
    let mut builder = CoverTreeBuilder::new();
    let verbosity = params["verbosity"].as_i64().unwrap_or(2) as u32;
    builder
        .set_scale_base(scale_base)
        .set_leaf_cutoff(leaf_cutoff)
        .set_min_res_index(min_res_index)
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
        params["leaf_cutoff"]
            .as_i64()
            .expect("Unable to read the 'leaf_cutoff'") as usize,
        params["min_res_index"]
            .as_i64()
            .expect("Unable to read the 'min_res_index'") as i32,
        params["use_singletons"]
            .as_bool()
            .expect("Unable to read the 'use_singletons'"),
    )
}

/// Helper function that handles the file I/O and protobuf decoding for you.
pub fn load_tree<P: AsRef<Path>, M: Metric>(
    tree_path: P,
    point_cloud: PointCloud<M>,
) -> GrandmaResult<CoverTreeWriter<M>> {
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
) -> GrandmaResult<()> {
    let tree_path_ref: &Path = tree_path.as_ref();

    println!("Saving tree to : {}", tree_path_ref.to_string_lossy());
    if tree_path_ref.exists() {
        let tree_path_str = match tree_path_ref.to_str() {
            Some(expr) => expr,
            None => panic!("Unicode error with the tree path"),
        };
        println!("\t \t {:?} exists, removing", tree_path_str);
        remove_file(&tree_path).map_err(GrandmaError::from)?;
    }

    let cover_proto = cover_tree.save();

    let mut core_file = OpenOptions::new()
        .read(true)
        .write(true)
        .create(true)
        .open(&tree_path).unwrap();

    let mut cos = CodedOutputStream::new(&mut core_file);
    cover_proto
        .write_to(&mut cos)
        .map_err(GrandmaError::from)?;
    cos.flush().map_err(GrandmaError::from)?;
    Ok(())
}
