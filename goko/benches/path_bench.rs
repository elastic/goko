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

use goko::*;
use pointcloud::*;
use pointcloud::{data_sources::*, label_sources::*, loaders::*};
use std::path::Path;

use std::sync::Arc;

use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn build_tree() -> CoverTreeWriter<SimpleLabeledCloud<DataRam<L2>, SmallIntLabels>> {
    let file_name = "data/ember_complex.yml";
    let path = Path::new(file_name);
    if !path.exists() {
        panic!("data/ember_complex.yml does not exist");
    }
    let builder = CoverTreeBuilder::from_yaml(&path);
    let point_cloud = labeled_ram_from_yaml("data/ember_complex.yml").unwrap();
    builder.build(Arc::new(point_cloud)).unwrap()
}

pub fn criterion_benchmark(c: &mut Criterion) {
    let ct = build_tree();
    let reader = ct.reader();
    c.bench_function("Known Path 0", |b| {
        b.iter(|| reader.known_path(black_box(0)))
    });

    let pointcloud = reader.point_cloud();
    let point = pointcloud.point(0).unwrap();
    c.bench_function("Unknown Path 0", |b| {
        b.iter(|| reader.path(black_box(&point)))
    });
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);
