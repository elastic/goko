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

extern crate protobuf;
extern crate rand;
extern crate yaml_rust;
use std::path::Path;
#[allow(dead_code)]
extern crate goko;
extern crate pointcloud;
use goko::*;
use pointcloud::*;
use pointcloud::{data_sources::*, label_sources::*, loaders::*};

use std::collections::HashMap;
use std::sync::Arc;
use std::time;

use goko::query_interface::BulkInterface;

fn build_tree() -> CoverTreeWriter<SimpleLabeledCloud<DataRam<L2>, SmallIntLabels>> {
    let file_name = "data/ember_complex.yml";
    let path = Path::new(file_name);
    if !path.exists() {
        panic!(file_name.to_owned() + &" does not exist".to_string());
    }
    let builder = CoverTreeBuilder::from_yaml(&path);
    let point_cloud = labeled_ram_from_yaml("data/ember_complex.yml").unwrap();
    builder.build(Arc::new(point_cloud)).unwrap()
}

fn main() {
    let mut ct = build_tree();
    ct.generate_summaries();
    let test_set = build_test_set();
    ct.refresh();
    let ct_reader = ct.reader();
    println!("Tree has {} nodes", ct_reader.node_count());

    let start = time::Instant::now();
    let bulk = BulkInterface::new(ct.reader());
    let point_indexes = ct_reader.point_cloud().reference_indexes();
    let tau = 0.05;
    let depths = bulk.known_path_and(&point_indexes, |reader,path| 
        if let Ok(path) = path {
            let mut homogenity_depth = path.len();
            for (i, (_d, a)) in path.iter().enumerate() {
                let summ = reader.get_node_label_summary(*a).unwrap();
                if summ.summary.items.len() == 1 {
                    homogenity_depth = i;
                    break;
                }
                let sum = summ.summary.items.iter().map(|(_, c)| c).sum::<usize>() as f32;
                let max = *summ.summary.items.iter().map(|(_, c)| c).max().unwrap() as f32;
                if 1.0 - max / sum < tau {
                    homogenity_depth = i;
                    break;
                }
            }
            (path.len(), homogenity_depth)
        } else {
            (0, 0)
        }
    );

    let mut final_depths = HashMap::new();
    for (f,h) in &depths {
        final_depths.entry(f-h).and_modify(|c| *c += 1).or_insert(1);
    }
    let mut keys: Vec<usize> = final_depths.keys().cloned().collect();
    keys.sort();
    println!("Final Depths:");
    for k in keys {
        println!("{}: {:?}", k, final_depths.get(&k).unwrap());
    }

    let elapse = start.elapsed().as_millis();
    println!(
        "Time elapsed {:?} milliseconds, time per sequence {} milliseconds",
        elapse,
        (elapse as f64) / (depths.len() as f64)
    );
}
