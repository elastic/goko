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
use goko::plugins::discrete::prelude::*;
use pointcloud::*;
use pointcloud::{data_sources::*, label_sources::*, loaders::*};

use std::collections::HashMap;
use std::sync::Arc;
use std::time;

use goko::query_interface::BulkInterface;

fn build_tree() -> CoverTreeWriter<SimpleLabeledCloud<DataRam<L2>, SmallIntLabels>> {
    let file_name = "../data/sorel_validation_memmap.yml";
    let path = Path::new(file_name);
    if !path.exists() {
        panic!("{} does not exist", file_name);
    }
    let builder = CoverTreeBuilder::from_yaml(&path);
    let point_cloud = labeled_ram_from_yaml("../data/sorel_validation_memmap.yml").unwrap();
    builder.build(Arc::new(point_cloud)).unwrap()
}

fn main() {
    let mut ct = build_tree();
    let start = time::Instant::now();
    ct.generate_summaries();
    ct.add_plugin::<GokoDirichlet>(GokoDirichlet {});
    ct.refresh();

    let elapse = start.elapsed().as_millis();
    println!(
        "Time elapsed {:?} milliseconds",
        elapse,
    );
}
