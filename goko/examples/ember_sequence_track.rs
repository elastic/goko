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

use goko::plugins::distributions::*;
use rand::prelude::*;

use rayon::prelude::*;

fn build_tree() -> CoverTreeWriter<SimpleLabeledCloud<DataRam<L2>, VecLabels>> {
    let file_name = "../data/ember_complex.yml";
    let path = Path::new(file_name);
    if !path.exists() {
        panic!(file_name.to_owned() + &" does not exist".to_string());
    }
    let builder = CoverTreeBuilder::from_yaml(&path);
    let point_cloud = vec_labeled_ram_from_yaml("../data/ember_complex_test.yml").unwrap();
    builder.build(Arc::new(point_cloud)).unwrap()
}

fn build_test_set() -> SimpleLabeledCloud<DataRam<L2>, VecLabels> {
    vec_labeled_ram_from_yaml("../data/ember_complex_test.yml").unwrap()
}

fn test_run(
    mut tracker: BayesCategoricalTracker<SimpleLabeledCloud<DataRam<L2>, VecLabels>>,
    test_set: &SimpleLabeledCloud<DataRam<L2>, VecLabels>,
    count: usize,
) -> Vec<KLDivergenceStats> {
    let mut rng = thread_rng();
    let mut stats = Vec::new();
    for i in 0..count {
        let point_index: usize = rng.gen_range(0, test_set.len());
        let point = test_set.point(point_index).unwrap();
        let trace = tracker.tree_reader().path(point).unwrap();
        tracker.add_path(trace);
        if i % 5 == 0 {
            stats.push(tracker.current_stats());
        }
    }
    stats
}

fn main() {
    let mut ct = build_tree();
    ct.add_plugin::<GokoDirichlet>(DirichletTree {});
    let test_set = build_test_set();
    //ct.cluster().unwrap();
    ct.refresh();
    let ct_reader = ct.reader();
    println!("Tree has {} nodes", ct_reader.node_count());
    let prior_weight = 1.0;
    let observation_weight = 1.3;
    let window_size = 40;
    let sequence_len = 1000;

    let trackers: Vec<BayesCategoricalTracker<SimpleLabeledCloud<DataRam<L2>, VecLabels>>> = (0
        ..1000)
        .map(|_| {
            BayesCategoricalTracker::new(prior_weight, observation_weight, window_size, ct.reader())
        })
        .collect();

    let start = time::Instant::now();
    let _stats: Vec<Vec<KLDivergenceStats>> = trackers
        .into_par_iter()
        .map(|tracker| test_run(tracker, &test_set, sequence_len))
        .collect();
    let elapse = start.elapsed().as_secs();
    println!(
        "Time elapsed {:?}, time per sequence {}",
        elapse,
        (elapse as f64) / 1000.0
    );
}
