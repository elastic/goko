use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

use core_goko::*;

pub fn criterion_benchmark(c: &mut Criterion) {
    let scale_index = -5;
    let point_index = 124;
    let na = NodeAddress::from((scale_index, point_index));
    c.bench_function("From", |b| {
        b.iter(||  NodeAddress::from(black_box((scale_index, point_index)))   )
    });

    c.bench_function("new_unsafe", |b| {
        b.iter(||  unsafe { NodeAddress::new_unchecked(black_box(scale_index), black_box(point_index)) }   )
    });

    c.bench_function("scale_index", |b| {
        b.iter(||  black_box(na).scale_index())
    });

    c.bench_function("point_index", |b| {
        b.iter(||  black_box(na).point_index())
    });
}

fn address_to_vec(c: &mut Criterion) {
    let mut group = c.benchmark_group("address_to_vec");
    for size in [2usize, 4, 8, 16, 32, 64, 128, 256].iter() {
        let addresses: Vec<NodeAddress> = (0..*size).map(|s| (s as i32, s).into()).collect();
        group.bench_with_input(
            BenchmarkId::new("address_to_point_index", size),
            &addresses,
            |b, addresses| {
                b.iter(|| addresses.to_point_indexes());
            },
        );
        group.bench_with_input(
            BenchmarkId::new("address_to_scale_index", size),
            &addresses,
            |b, addresses| {
                b.iter(|| addresses.to_scale_indexes());
            },
        );
    }
    group.finish();
}

criterion_group!(benches,criterion_benchmark, address_to_vec);
criterion_main!(benches);