use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use core_goko::*;
use stats_goko::discrete::{Categorical, Dirichlet, DirichletTracker, DiscreteData};

pub fn criterion_benchmark(c: &mut Criterion) {
    let params_1: Vec<(u64, f64)> = (0..10).map(|i| (i, 6.0)).collect();
    let bucket1 = Categorical::from(&params_1[..]);
    let params_2: Vec<(u64, f64)> = (0..10).map(|i| (i, i as f64)).collect();
    let bucket2 = Categorical::from(&params_2[..]);
    let params_3: Vec<(u64, f64)> = (0..3).map(|i| (i * 3, i as f64)).collect();

    c.bench_function("Categorical KL Div", |b| {
        b.iter(|| bucket1.kl_div(black_box(&bucket2)))
    });

    c.bench_function("Categorical Supported KL Div", |b| {
        b.iter(|| bucket1.supported_kl_div(black_box(&bucket2)))
    });

    let diri_bucket1: Dirichlet = params_1[..].into();
    let data_bucket2: DiscreteData = params_2[..].into();
    let data_bucket3: DiscreteData = params_3[..].into();
    c.bench_function("Dirichlet posterior KL Div Equal Size", |b| {
        b.iter(|| diri_bucket1.posterior_kl_div(black_box(&data_bucket2)))
    });
    c.bench_function("Dirichlet posterior KL Div Smaller", |b| {
        b.iter(|| diri_bucket1.posterior_kl_div(black_box(&data_bucket3)))
    });

    let mut diri_bucket2: Dirichlet = params_2[..].into();
    diri_bucket2.add_observations(&data_bucket2);
    c.bench_function("Supported Dirichlet KL Div", |b| {
        b.iter(|| diri_bucket1.supported_kl_div(black_box(&diri_bucket2)))
    });
    c.bench_function("Dirichlet KL Div", |b| {
        b.iter(|| diri_bucket1.kl_div(black_box(&diri_bucket2)))
    });
}

fn tracker_fast(tracker: &mut DirichletTracker, observations: &[NodeAddress]) {
    for (i, observation) in observations.iter().enumerate() {
        tracker.add_observation(*observation);
        if i % 10 == 0 {
            tracker.mll();
            tracker.kl_div();
        }
    }
}

fn tracker(c: &mut Criterion) {
    let params: Vec<(u64, f64)> = (0..10).map(|i| (i, 6.0)).collect();
    let prior: Dirichlet = (&params[..]).into();
    let mut group = c.benchmark_group("tracker");
    for size in [2u64, 32, 512, 512 * 16, 512 * 16 * 16].iter() {
        let mut tracker: DirichletTracker = prior.tracker();
        let observations: Vec<NodeAddress> =
            (0..*size).map(|s| NodeAddress::from(s % 10)).collect();
        group.bench_with_input(
            BenchmarkId::new("tracker_fast", size),
            &observations,
            |b, observations| {
                b.iter(|| tracker_fast(&mut tracker, &observations));
            },
        );

        let wide_params: Vec<(u64, f64)> = (0..*size).map(|i| (i, 6.0)).collect();
        let wide_prior: Dirichlet = (&wide_params[..]).into();
        let mut wide_tracker: DirichletTracker = wide_prior.tracker();
        group.bench_with_input(
            BenchmarkId::new("tracker_wide", size),
            &observations,
            |b, observations| {
                b.iter(|| tracker_fast(&mut wide_tracker, &observations));
            },
        );
    }
    group.finish();
}

fn dirichlet(c: &mut Criterion) {
    let mut group = c.benchmark_group("dirichlet ln_pdf");
    for size in [2u64, 2 * 16, 2 * 16 * 16, 2 * 16 * 16 * 16, 512 * 16 * 16].iter() {
        let params: Vec<(u64, f64)> = (0..*size).map(|i| (i, 6.0)).collect();
        let prior: Dirichlet = Dirichlet::from(&params[..]);
        let categorical: Categorical = Categorical::from(&params[..]);
        let multinomial: DiscreteData = DiscreteData::from(&params[..]);

        group.bench_with_input(
            BenchmarkId::new("categorical size", size),
            &categorical,
            |b, categorical| {
                b.iter(|| prior.ln_pdf(&categorical));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("multinomial size", size),
            &multinomial,
            |b, multinomial| {
                b.iter(|| prior.ln_likelihood(&multinomial));
            },
        );

        let params: Vec<(u64, f64)> = (0..64).map(|_| (*size, 6.0)).collect();
        let prior: Dirichlet = Dirichlet::from(&params[..]);
        let categorical: Categorical = Categorical::from(&params[..]);
        let multinomial: DiscreteData = DiscreteData::from(&params[..]);
        group.bench_with_input(
            BenchmarkId::new("categorical count", size),
            &categorical,
            |b, categorical| {
                b.iter(|| prior.ln_pdf(&categorical));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("multinomial count", size),
            &multinomial,
            |b, multinomial| {
                b.iter(|| prior.ln_likelihood(&multinomial));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark, tracker, dirichlet);
criterion_main!(benches);
