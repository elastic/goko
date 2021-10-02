use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use stats_plugins::Categorical;
use stats_plugins::{Dirichlet, DirichletTracker};

pub fn criterion_benchmark(c: &mut Criterion) {
    let mut bucket1 = Categorical::new();
    for i in 0..10 {
        bucket1.add_pop(i, 6.0);
    }
    println!("{:?}", bucket1);

    let mut bucket2 = Categorical::new();
    for i in 0..10 {
        bucket2.add_pop(i, i as f64);
    }

    let mut bucket3 = Categorical::new();
    for i in 0..3 {
        bucket3.add_pop(i * 3, i as f64);
    }

    c.bench_function("Categorical KL Divergence", |b| {
        b.iter(|| bucket1.kl_divergence(black_box(&bucket2)))
    });

    c.bench_function("Categorical Supported KL Divergence", |b| {
        b.iter(|| bucket1.supported_kl_divergence(black_box(&bucket2)))
    });

    let diri_bucket1: Dirichlet = bucket1.into();
    c.bench_function("Dirichlet posterior KL Divergence Equal Size", |b| {
        b.iter(|| diri_bucket1.posterior_kl_divergence(black_box(&bucket2)))
    });
    c.bench_function("Dirichlet posterior KL Divergence Smaller", |b| {
        b.iter(|| diri_bucket1.posterior_kl_divergence(black_box(&bucket3)))
    });

    let mut diri_bucket2: Dirichlet = diri_bucket1.clone();
    diri_bucket2.add_evidence(&bucket2);
    c.bench_function("Supported Dirichlet KL Divergence", |b| {
        b.iter(|| diri_bucket1.supported_kl_divergence(black_box(&diri_bucket2)))
    });
    c.bench_function("Dirichlet KL Divergence", |b| {
        b.iter(|| diri_bucket1.kl_divergence(black_box(&diri_bucket2)))
    });
}

fn tracker_fast(tracker: &mut DirichletTracker, observations: &[u64]) {
    for observation in observations {
        tracker.add_observation(*observation);
    }
}

fn tracker_slow(prior: &Dirichlet, categorical: &mut Categorical, observations: &[u64]) {
    for observation in observations {
        categorical.add_pop(*observation, 1.0);
        prior.posterior_kl_divergence(categorical);
    }
}

fn tracker(c: &mut Criterion) {
    let mut bucket1 = Categorical::new();
    for i in 0..10 {
        bucket1.add_pop(i, 6.0);
    }
    let prior: Dirichlet = bucket1.into();
    let mut group = c.benchmark_group("tracker");
    for size in [2u64, 4, 8, 16, 32].iter() {
        let mut tracker: DirichletTracker = prior.tracker();
        let mut categorical: Categorical = Categorical::new();
        let observations: Vec<u64> = (0..*size).map(|s| s % 10).collect();
        group.bench_with_input(
            BenchmarkId::new("tracker_fast", size),
            &observations,
            |b, observations| {
                b.iter(|| tracker_fast(&mut tracker, &observations));
            },
        );
        group.bench_with_input(
            BenchmarkId::new("tracker_slow", size),
            &observations,
            |b, observations| {
                b.iter(|| tracker_slow(&prior, &mut categorical, &observations));
            },
        );
    }
    group.finish();
}

criterion_group!(benches, criterion_benchmark, tracker);
criterion_main!(benches);
