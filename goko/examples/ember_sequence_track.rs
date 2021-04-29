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

use std::sync::Arc;
use std::time;

use goko::plugins::discrete::prelude::*;
use goko::query_interface::BulkInterface;

fn build_tree() -> CoverTreeWriter<SimpleLabeledCloud<DataRam<L2>, VecLabels>> {
    let file_name = "data/ember_complex.yml";
    let path = Path::new(file_name);
    if !path.exists() {
        panic!("{} does not exist", file_name);
    }
    let builder = CoverTreeBuilder::from_yaml(&path);
    let point_cloud = vec_labeled_ram_from_yaml("data/ember_complex.yml").unwrap();
    builder.build(Arc::new(point_cloud)).unwrap()
}

fn build_test_set() -> SimpleLabeledCloud<DataRam<L2>, VecLabels> {
    vec_labeled_ram_from_yaml("data/ember_complex_test.yml").unwrap()
}

fn main() {
    let mut ct = build_tree();
    ct.add_plugin::<GokoDirichlet>(GokoDirichlet {});
    let test_set = build_test_set();
    //ct.cluster().unwrap();
    ct.refresh();
    let ct_reader = ct.reader();
    println!("Tree has {} nodes", ct_reader.node_count());
    let window_size = 0;
    let num_sequence = 8;

    let mut baseline = DirichletBaseline::default();
    baseline.set_sequence_len(test_set.len());
    baseline.set_num_sequences(num_sequence);
    println!(
        "Gathering baseline with window_size: {}",
        window_size
    );
    let baseline_start = time::Instant::now();
    let _baseline_data = baseline.train(ct.reader());
    let baseline_elapse = baseline_start.elapsed().as_millis();

    println!(
        "BASELINE: Time elapsed {:?} milliseconds, time per sequence {} milliseconds",
        baseline_elapse,
        (baseline_elapse as f64) / ((test_set.len() * num_sequence) as f64)
    );

    let mut tracker =
        BayesCategoricalTracker::new( window_size, ct.reader());
    let start = time::Instant::now();

    let points: Vec<&[f32]> = (0..test_set.len())
        .map(|i| test_set.point(i).unwrap())
        .collect();
    let bulk = BulkInterface::new(ct.reader());
    let mut paths = bulk.path(&points);
    for path in paths.drain(0..) {
        tracker.add_path(path.unwrap());
    }

    let elapse = start.elapsed().as_millis();
    println!(
        "Time elapsed {:?} milliseconds, time per sequence {} milliseconds",
        elapse,
        (elapse as f64) / (test_set.len() as f64)
    );
    println!("stats: {:?}", tracker.kl_div_stats());
}
