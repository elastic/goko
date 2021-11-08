use goko::plugins::discrete::prelude::*;
use goko::NodeAddress;
use goko::{CoverTreeBuilder, CoverTreeReader, CoverTreeWriter};
use lightgbm::Booster;
use pointcloud::data_sources::DataMemmap;
use pointcloud::glued_data_cloud::HashGluedCloud;
use pointcloud::label_sources::VecLabels;
use pointcloud::loaders::open_labeled_memmaps;
use pointcloud::metrics::L2;
use pointcloud::*;
use rand::distributions::Standard;
use rand::{Rng, SeedableRng};
use rayon::prelude::*;
use serde_json;
use std::collections::HashMap;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::sync::Arc;
use std::time::Instant;
use rand::seq::SliceRandom;
use serde::Serialize;
type SorelCloud = HashGluedCloud<SimpleLabeledCloud<DataMemmap<L2>, VecLabels>>;

fn main() {
    let attack_rates = [0.0, 0.0001, 0.001, 0.01, 0.1, 1.0];
    let baseline_sample_rates = [1000, 5000, 25000];
    let sample_rates = [1000, 10000, 100000];
    let num_models = 5;
    let num_attacks = 10;
    let baseline_len = 60;
    let num_covertrees = 48;
    let leaf_cutoff = 500;

    // Opens a set of memmaps of both data and labels
    let validation_cloud = Arc::new(
        open_labeled_memmaps::<L2>(
            2381,
            1,
            &["/localdata/sorel/X_validation.dat".into()],
            &["/localdata/sorel/y_validation.dat".into()],
        )
        .unwrap(),
    );

    let test_cloud = open_labeled_memmaps::<L2>(
        2381,
        1,
        &["/localdata/sorel/X_test.dat".into()],
        &["/localdata/sorel/y_test.dat".into()],
    )
    .unwrap();

    let mut rng = rand::thread_rng();
    let mut overall_indexes: Vec<usize> = (0..test_cloud.len()).collect();
    overall_indexes.shuffle(&mut rng);
    let baseline_indexes = &overall_indexes[..(overall_indexes.len()/2)];
    let test_indexes = &overall_indexes[(overall_indexes.len()/2)..];

    let mut ct_builder = CoverTreeBuilder::default();
    ct_builder
        .set_leaf_cutoff(leaf_cutoff)
        .set_min_res_index(-2)
        .set_scale_base(1.3)
        .set_verbosity(2);
    (0..num_covertrees)
        .for_each(|i| {
            let ct_path = format!("/localdata/sorel/covertrees/tree_{}_{}.ct", leaf_cutoff, i);
            if !Path::new(&ct_path).exists() {
                println!("Building and saving {}", ct_path);
                let ct = ct_builder.build(Arc::clone(&validation_cloud)).unwrap();
                goko::utils::save_tree(&ct_path, &ct).unwrap();
                save_tree_maps(ct.reader(), leaf_cutoff, i);
            };
            
        });
    let covertrees: Vec<CoverTreeWriter<_>> = (0..num_covertrees).into_par_iter().map(|i| {
        let ct_path = format!("/localdata/sorel/covertrees/tree_{}_{}.ct", leaf_cutoff, i);
        let mut ct: CoverTreeWriter<SorelCloud> = goko::utils::load_tree(&ct_path, Arc::clone(&validation_cloud)).unwrap();
        ct.generate_summaries();
        ct.add_plugin::<GokoDirichlet>(GokoDirichlet {});
        ct
    }).collect();
    let ct_readers: Vec<CoverTreeReader<_>> = covertrees.iter().map(|ct| ct.reader()).collect();
    println!("Building Base Trackers");

    let tracker_creation_now = Instant::now();
    let base_trackers: Vec<BayesCovertree> = ct_readers.clone().into_par_iter().map(|covertree| BayesCovertree::new(0, &covertree)).collect();
    println!(
        "Finished trackers, took {:?} with {} per second",
        tracker_creation_now.elapsed(),
        (sample_rates.len() as f32) / tracker_creation_now.elapsed().as_secs_f32()
    );
    
    println!("Building baselines!");
    let baseline_now = Instant::now();
    ct_readers
        .clone()
        .into_par_iter()
        .enumerate()
        .zip(&base_trackers)
        .for_each(|((i, ct), base_tracker)| {
            run_test_set_baseline(base_tracker, &baseline_indexes, &test_cloud, &sample_rates, &baseline_sample_rates, ct, i, leaf_cutoff, baseline_len)
        });
    println!(
        "Finished baselines, took {:?} with {} per second",
        baseline_now.elapsed(),
        ((num_covertrees * baseline_len) as f32) / baseline_now.elapsed().as_secs_f32()
    );

    let attack_now = Instant::now();
    
    ct_readers
        .clone()
        .into_par_iter()
        .enumerate()
        .zip(&base_trackers)
        .for_each(|((i, ct), base_tracker)| {
            run_test_set_attack_set(
                base_tracker,
                test_indexes,
                &test_cloud,
                &attack_rates,
                &sample_rates,
                num_attacks,
                num_models,
                ct,
                i,
                leaf_cutoff,
            )
        });

    println!(
        "Finished attacks, took {:?} with {} per second",
        attack_now.elapsed(),
        ((num_covertrees * num_models * num_attacks) as f32) / attack_now.elapsed().as_secs_f32()
    );
}

fn save_tree_maps(ct: CoverTreeReader<SorelCloud>, leaf_cutoff: usize, i: usize) {
    let mut child_parent: HashMap<u64, u64> = HashMap::new();
    let mut node_address: HashMap<u64, (i32, usize)> = HashMap::new();
    for (_, layer) in ct.layers() {
        layer.for_each_node(|_, node| {
            if let Some(pa) = node.parent_address() {
                node_address.insert(pa.raw(), pa.to_tuple().unwrap());
                child_parent.insert(node.address().raw(), pa.raw());
            }
            node_address.insert(node.address().raw(), node.address().to_tuple().unwrap());
        });
    }
    let mut file = File::create(format!(
        "/localdata/sorel/covertrees/tree_{}_{}_child_parent.json",
        leaf_cutoff, i
    ))
    .unwrap();
    let json = serde_json::to_string(&child_parent).unwrap();
    file.write_all(json.as_bytes()).unwrap();
    let mut file = File::create(format!(
        "/localdata/sorel/covertrees/tree_{}_{}_node_address.json",
        leaf_cutoff, i
    ))
    .unwrap();
    let json = serde_json::to_string(&node_address).unwrap();
    file.write_all(json.as_bytes()).unwrap();
}

fn run_test_set_baseline(
    base_tracker: &BayesCovertree,
    baseline_indexes: &[usize],
    test_cloud: &SorelCloud,
    sample_rates: &[usize],
    observation_rates: &[usize],
    covertree: CoverTreeReader<SorelCloud>,
    covertree_index: usize,
    leaf_cutoff: usize,
    baseline_len: usize,
) {
    let mut rng = rand::rngs::SmallRng::from_entropy();
    let subsample_rates: Vec<usize> = sample_rates
        .iter()
        .filter(|sr| {
            let baseline_path = format!(
                "/localdata/sorel/covertrees/test_set_baselines/tree_{}_{}_baseline_{}.json",
                leaf_cutoff, covertree_index, sr
            );
            !Path::new(&baseline_path).exists()
        })
        .copied()
        .collect();

    let mut baseline_builders: Vec<BayesCovertreeBaselineBuilder> = subsample_rates
        .iter()
        .zip(observation_rates)
        .map(|(sr, oi)| BayesCovertreeBaselineBuilder::new(*sr,*oi, base_tracker.clone_new_window(*sr)))
        .collect();

    for _i in 0..(*subsample_rates.iter().max().unwrap_or(&0) * baseline_len) {
        let test_index = baseline_indexes.choose(&mut rng).unwrap();
        let test_point = test_cloud.point(*test_index).unwrap();
        let test_path = covertree.path(&test_point).unwrap();
        for baseline_builder in baseline_builders.iter_mut() {
            baseline_builder.add_path(test_path.clone());
        }
    }

    for (sr, baseline_builder) in sample_rates.iter().zip(&baseline_builders) {
        save_to_file(&baseline_builder.baseline(), &format!(
            "/localdata/sorel/covertrees/test_set_baselines/tree_{}_{}_baseline_{}.json",
            leaf_cutoff, covertree_index, sr
        ));
        save_to_file(&baseline_builder.loo_violators(), &format!(
            "/localdata/sorel/covertrees/test_set_baselines/tree_{}_{}_baseline_{}_loo_violators.json",
            leaf_cutoff, covertree_index, sr
        ));
    }
}

fn run_test_set_attack_set(
    base_tracker: &BayesCovertree,
    test_indexes: &[usize],
    test_cloud: &SorelCloud,
    attack_rates: &[f32],
    sample_rates: &[usize],
    num_attacks: usize,
    num_models: usize,
    covertree: CoverTreeReader<SorelCloud>,
    covertree_index: usize,
    leaf_cutoff: usize,
) {
    let bst: Vec<Booster> = (0..num_models).map(|model_index|Booster::from_file(&format!(
        "/localdata/sorel/lightGBM/seed{}/lightgbm.model",
        model_index
    ))
    .unwrap()).collect();
    for attack_index in 0..num_attacks {
        for (model_index, model) in bst.iter().enumerate() {
            let trackers: Vec<BayesCovertree> = attack_rates
                .iter()
                .map(|_| base_tracker.clone())
                .collect();
            run_test_set_attack(trackers, test_indexes,test_cloud,attack_rates, sample_rates,&covertree, model, model_index, covertree_index, leaf_cutoff, attack_index);
        }   
    }
}

fn run_test_set_attack(
    mut trackers: Vec<BayesCovertree>,
    test_indexes: &[usize],
    test_cloud: &SorelCloud,
    attack_rates: &[f32],
    sample_rates: &[usize],
    covertree: &CoverTreeReader<SorelCloud>,
    model: &Booster,
    model_index: usize,
    covertree_index: usize,
    leaf_cutoff: usize,
    attack_index: usize,
) {
    let mut attack_path: Option<Vec<(NodeAddress, f32)>> = None;
    let mut rng = rand::rngs::SmallRng::from_entropy();
    while attack_path.is_none() {
        let test_index = test_indexes.choose(&mut rng).unwrap();
        let test_point_ref = test_cloud.point(*test_index).unwrap();
        let test_path = covertree.path(&test_point_ref).unwrap();
        let test_point: Vec<f64> = test_point_ref.iter().map(|v| *v as f64).collect();
        let test_point_label = test_cloud.label(*test_index).unwrap().unwrap()[0];
        let pred = model.predict(vec![test_point]).unwrap()[0][0];
        if pred < 0.5 && test_point_label != 0.0 {
            attack_path = Some(test_path.clone());
        }
    }

    for i in 0..*sample_rates.iter().max().unwrap_or(&0) {
        let test_index = test_indexes.choose(&mut rng).unwrap();
        let test_point = test_cloud.point(*test_index).unwrap();
        let test_path = covertree.path(&test_point).unwrap();
        let attack_roll: f32 = rng.sample(Standard);
        for (ar, tracker) in attack_rates.iter().zip(trackers.iter_mut()) {
            if attack_path.is_some() && attack_roll < *ar {
                tracker.add_path(attack_path.clone().unwrap());
            } else {
                tracker.add_path(test_path.clone());
            }
            if sample_rates.contains(&(i + 1)) {
                save_to_file(&tracker.small(),
                    &format!("/localdata/sorel/covertrees/test_set_attack_results/model_{}_tree_{}_{}_attack_{}_{}_{}.json",
                        model_index,
                        leaf_cutoff,
                        covertree_index,
                        attack_index,
                        ar,
                        i+1)
                );
            }
        }
    }
    let path: Vec<u64> = attack_path
        .map(|p| p.iter().map(|(na, _)| na.raw()).collect())
        .unwrap_or(Vec::new());
    save_to_file(&path,
        &format!("/localdata/sorel/covertrees/test_set_attack_results/model_{}_tree_{}_{}_attack_{}_attack_path.json",
            model_index,
            leaf_cutoff,
            covertree_index, 
            attack_index)
    );
}

fn save_to_file<S: Serialize>(s: &S, file_name: &str) {
    println!("Saving : {}", file_name);
    let mut file = File::create(file_name).unwrap();
    let json = serde_json::to_string(s).unwrap();
    file.write_all(json.as_bytes()).unwrap();
}