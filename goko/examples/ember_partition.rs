use goko::*;
use pointcloud::*;
use pointcloud::{data_sources::*, label_sources::*, loaders::*};
use std::path::Path;
use std::time::Instant;

use std::sync::Arc;


fn build_tree(seed: u64, partition_type: PartitionType, leaf_cutoff:usize) -> (f32,CoverTreeWriter<SimpleLabeledCloud<DataRam<L2>, SmallIntLabels>>) {
    let file_name = "../data/ember_complex.yml";
    let path = Path::new(file_name);
    if !path.exists() {
        panic!(file_name.to_owned() + &" does not exist".to_string());
    }
    let mut builder = CoverTreeBuilder::from_yaml(&path);
    builder.set_partition_type(partition_type);
    builder.set_rng_seed(seed);
    builder.set_leaf_cutoff(leaf_cutoff);
    let point_cloud = labeled_ram_from_yaml("../data/ember_complex.yml").unwrap();
    
    let start = Instant::now();
    let ct = builder.build(Arc::new(point_cloud)).unwrap();
    let end = Instant::now();
    ((end - start).as_secs_f32(), ct)
}

#[derive(Debug,Default)]
struct QueryRunStats {
    knn_moment1: f32, 
    knn_moment2: f32,
    path_moment1: f32, 
    path_moment2: f32,
    routing_knn_moment1: f32, 
    routing_knn_moment2: f32,
    count: usize,
}

fn run(seed: u64, query_count: usize, partition_type:PartitionType, leaf_cutoff:usize) -> (f32,QueryRunStats) {
    let mut query_stats = QueryRunStats::default();
    let (build_time,ct) = build_tree(seed,partition_type,leaf_cutoff);
    let reader = ct.reader();
    let pointcloud = reader.point_cloud();
    for i in 0..query_count {
        let point = pointcloud.point(i).unwrap();

        let start = Instant::now();
        reader.path(&point).unwrap();
        let end = Instant::now();
        let time_taken: f32 = (end - start).as_secs_f32();
        query_stats.path_moment1 += time_taken;
        query_stats.path_moment2 += time_taken*time_taken;

        let start = Instant::now();
        reader.knn(&point,10).unwrap();
        let end = Instant::now();
        let time_taken: f32 = (end - start).as_secs_f32();
        query_stats.knn_moment1 += time_taken;
        query_stats.knn_moment2 += time_taken*time_taken;

        let start = Instant::now();
        reader.routing_knn(&point,10).unwrap();
        let end = Instant::now();
        let time_taken: f32 = (end - start).as_secs_f32();
        query_stats.routing_knn_moment1 += time_taken;
        query_stats.routing_knn_moment2 += time_taken*time_taken;

        query_stats.count += 1;
    }
    (build_time,query_stats)
} 

fn run_group(tree_iterations: usize, query_iterations: usize, partition_type:PartitionType, leaf_cutoff:usize) -> () {
    let mut first_query_stats = QueryRunStats::default();
    let mut build_time_moment1: f32 = 0.0;
    let mut build_time_moment2: f32 = 0.0;
    for seed in 0..tree_iterations {
        let (run_build_time, run_stats) = run(seed as u64,query_iterations,partition_type,leaf_cutoff);
        build_time_moment1 += run_build_time;
        build_time_moment2 += run_build_time*run_build_time;
        first_query_stats.knn_moment1 += run_stats.knn_moment1;
        first_query_stats.knn_moment2 += run_stats.knn_moment2;
        first_query_stats.path_moment1 += run_stats.path_moment1;
        first_query_stats.path_moment2 += run_stats.path_moment2;
        first_query_stats.routing_knn_moment1 += run_stats.routing_knn_moment1;
        first_query_stats.routing_knn_moment2 += run_stats.routing_knn_moment2;
        first_query_stats.count += run_stats.count;
    }
    println!("Parameters: leaf cutoff {}, partition type {:?}",leaf_cutoff, partition_type);
    let run_build_mean = build_time_moment1/(tree_iterations as f32);
    println!("\tRun mean build time: {}, run variance: {}", run_build_mean, build_time_moment2/(tree_iterations as f32) - run_build_mean*run_build_mean);

    let run_knn_mean = first_query_stats.knn_moment1/(first_query_stats.count as f32);
    println!("\tMean KNN query time: {}, variance: {}", run_knn_mean, first_query_stats.knn_moment2/(first_query_stats.count as f32) - run_knn_mean*run_knn_mean);
    let run_path_mean = first_query_stats.path_moment1/(first_query_stats.count as f32);
    println!("\tMean path query time: {}, variance: {}", run_path_mean, first_query_stats.path_moment2/(first_query_stats.count as f32) - run_path_mean*run_path_mean);
    let run_routing_knn_mean = first_query_stats.routing_knn_moment1/(first_query_stats.count as f32);
    println!("\tMean routing_knn query time: {}, variance: {}", run_routing_knn_mean, first_query_stats.routing_knn_moment2/(first_query_stats.count as f32) - run_routing_knn_mean*run_routing_knn_mean);
}


fn main() {
    let tree_iterations: usize = 5;
    let query_iterations: usize = 100;
    let leaf_cutoffs = [0,10,50,100];
    for leaf_cutoff in &leaf_cutoffs {
        run_group(tree_iterations,query_iterations,PartitionType::First, *leaf_cutoff);
        run_group(tree_iterations,query_iterations,PartitionType::Nearest, *leaf_cutoff);
    }
}