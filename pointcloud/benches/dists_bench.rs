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


use pointcloud::*;
use pointcloud::data_sources::*;
use pointcloud::glued_data_cloud::*;


use criterion::{black_box, criterion_group, criterion_main, Criterion};

fn build_ram_random_test<M:Metric>(count: usize, data_dim: usize) -> DataRam<M> {
    DataRam::<M>::new(
        (0..count * data_dim)
            .map(|_i| rand::random::<f32>())
            .collect(),
        data_dim,
    )
    .unwrap()
}


fn l2_benchmarks(c: &mut Criterion) {
    let count = 100;
    let dim = 303;
    let pc = build_ram_random_test::<L2>(count, dim);

    let indexes_small: [PointIndex; 9] = [1, 3, 5, 7, 9, 11, 13, 15, 17];
    let indexes_large: Vec<PointIndex> = (0..count).collect();

    let point = Point::Dense(vec![0.0; dim]);

    c.bench_function("L2_adjacency_matrix_small", |b| b.iter(|| pc.adjacency_matrix(black_box(&indexes_small)).unwrap()));
    c.bench_function("L2_adjacency_matrix_large", |b| b.iter(|| pc.adjacency_matrix(black_box(&indexes_large)).unwrap()));

    c.bench_function("L2_distances_to_point_small", |b| b.iter(|| pc.distances_to_point(black_box(&point),black_box(&indexes_small)).unwrap()));
    c.bench_function("L2_distances_to_point_large", |b| b.iter(|| pc.distances_to_point(black_box(&point),black_box(&indexes_large)).unwrap()));
}

fn l1_benchmarks(c: &mut Criterion) {
    let count = 100;
    let dim = 303;
    let pc = build_ram_random_test::<L1>(count, dim);

    let indexes_small: [PointIndex; 9] = [1, 3, 5, 7, 9, 11, 13, 15, 17];
    let indexes_large: Vec<PointIndex> = (0..count).collect();

    let point = Point::Dense(vec![0.0; dim]);

    c.bench_function("L1_adjacency_matrix_small", |b| b.iter(|| pc.adjacency_matrix(black_box(&indexes_small)).unwrap()));
    c.bench_function("L1_adjacency_matrix_large", |b| b.iter(|| pc.adjacency_matrix(black_box(&indexes_large)).unwrap()));

    c.bench_function("L1_distances_to_point_small", |b| b.iter(|| pc.distances_to_point(black_box(&point),black_box(&indexes_small)).unwrap()));
    c.bench_function("L1_distances_to_point_large", |b| b.iter(|| pc.distances_to_point(black_box(&point),black_box(&indexes_large)).unwrap()));
}

fn cosine_benchmarks(c: &mut Criterion) {
    let count = 100;
    let dim = 303;
    let pc = build_ram_random_test::<CosineSim>(count, dim);

    let indexes_small: [PointIndex; 9] = [1, 3, 5, 7, 9, 11, 13, 15, 17];
    let indexes_large: Vec<PointIndex> = (0..count).collect();

    let point = Point::Dense(vec![0.0; dim]);

    c.bench_function("CosineSim_adjacency_matrix_small", |b| b.iter(|| pc.adjacency_matrix(black_box(&indexes_small)).unwrap()));
    c.bench_function("CosineSim_adjacency_matrix_large", |b| b.iter(|| pc.adjacency_matrix(black_box(&indexes_large)).unwrap()));

    c.bench_function("CosineSim_distances_to_point_small", |b| b.iter(|| pc.distances_to_point(black_box(&point),black_box(&indexes_small)).unwrap()));
    c.bench_function("CosineSim_distances_to_point_large", |b| b.iter(|| pc.distances_to_point(black_box(&point),black_box(&indexes_large)).unwrap()));
}

fn small_glue_benchmarks(c: &mut Criterion) {
    let count = 100;
    let glue_count = 10;
    let dim = 303;
    let pc = HashGluedCloud::new((0..glue_count).map(|_| build_ram_random_test::<L2>(count, dim)).collect());

    let indexes_small: [PointIndex; 10] = [0, 10, 30, 50, 70, 90, 110, 130, 150, 170];
    let indexes_large: Vec<PointIndex> = (0..100).map(|i| i*5).collect();

    let point = Point::Dense(vec![0.0; dim]);

    c.bench_function("small_glue_distances_to_point_small", |b| b.iter(|| pc.distances_to_point(black_box(&point),black_box(&indexes_small)).unwrap()));
    c.bench_function("small_glue_distances_to_point_large", |b| b.iter(|| pc.distances_to_point(black_box(&point),black_box(&indexes_large)).unwrap()));
}

fn glue_benchmarks(c: &mut Criterion) {
    let count = 10;
    let glue_count = 100;
    let dim = 303;
    let pc = HashGluedCloud::new((0..glue_count).map(|_| build_ram_random_test::<L2>(count, dim)).collect());

    let indexes_small: [PointIndex; 10] = [0, 10, 30, 50, 70, 90, 110, 130, 150, 170];
    let indexes_large: Vec<PointIndex> = (0..glue_count).map(|i| i*5).collect();

    let point = Point::Dense(vec![0.0; dim]);

    c.bench_function("glue_distances_to_point_small", |b| b.iter(|| pc.distances_to_point(black_box(&point),black_box(&indexes_small)).unwrap()));
    c.bench_function("glue_distances_to_point_large", |b| b.iter(|| pc.distances_to_point(black_box(&point),black_box(&indexes_large)).unwrap()));
}

fn large_glue_benchmarks(c: &mut Criterion) {
    let count = 10;
    let glue_count = 65000;
    let dim = 303;
    let pc = HashGluedCloud::new((0..glue_count).map(|_| build_ram_random_test::<L2>(count, dim)).collect());

    let indexes_small: [PointIndex; 10] = [0, 10, 30, 50, 70, 90, 110, 130, 150, 170];
    let indexes_large: Vec<PointIndex> = (0..100).map(|i| i*5).collect();

    let point = Point::Dense(vec![0.0; dim]);

    c.bench_function("large_glue_distances_to_point_small", |b| b.iter(|| pc.distances_to_point(black_box(&point),black_box(&indexes_small)).unwrap()));
    c.bench_function("large_glue_distances_to_point_large", |b| b.iter(|| pc.distances_to_point(black_box(&point),black_box(&indexes_large)).unwrap()));
}

criterion_group!(benches, l1_benchmarks,l2_benchmarks,cosine_benchmarks, small_glue_benchmarks, glue_benchmarks, large_glue_benchmarks);
criterion_main!(benches);