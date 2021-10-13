//! # DiscreteData Distribution
//!
//! Simple probability distribution that enables you to simulated the rough
//! distribution of data in the tree.
use super::parameter_store::DiscreteParams;

use rand::distributions::{Distribution, Uniform};
use rand::Rng;

use super::dirichlet::Dirichlet;

/// Simple probability density function for where things go by count
/// Stored as a flat vector in the order of the node addresses.
#[derive(Debug, Clone, Default)]
pub struct DiscreteData {
    pub(crate) params: DiscreteParams,
}

impl DiscreteData {
    /// Creates a new empty bucket probability
    pub fn new() -> DiscreteData {
        DiscreteData {
            params: DiscreteParams::new(),
        }
    }

    /// Total input to this DiscreteData distribution.
    pub fn total(&self) -> f64 {
        self.params.total()
    }

    pub fn merge(&mut self, other: &DiscreteData) {
        for (na, c) in other.params.iter() {
            self.params.add_pop(na, c);
        }
    }

    pub fn add_pop(&mut self, loc: u64, count: f64) -> f64 {
        self.params.add_pop(loc, count)
    }

    pub fn remove_pop(&mut self, loc: u64, count: f64) -> f64 {
        self.params.remove_pop(loc, count)
    }

    pub fn get(&self, loc: u64) -> Option<f64> {
        self.params.get(loc)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    //use crate::tree::tests::build_basic_tree;

    #[test]
    fn empty_bucket_sanity_test() {
        let mut buckets = DiscreteData::new();
        assert_eq!(buckets.get(0), None);
        assert_eq!(buckets.total(), 0.0);
        buckets.add_pop(0, 5.0);
        assert_eq!(buckets.get(0), Some(5.0));
        assert_eq!(buckets.total(), 5.0);
    }

    #[test]
    fn merge_test() {
        let mut bucket1 = DiscreteData::new();
        bucket1.add_pop(0, 6.0);
        bucket1.add_pop(1, 6.0);
        println!("{:?}", bucket1);

        let mut bucket2 = DiscreteData::new();
        bucket2.add_pop(0, 4.0);
        bucket2.add_pop(1, 8.0);
        println!("{:?}", bucket2);
        bucket1.merge(&bucket2);
        assert_approx_eq!(bucket1.get(0).unwrap(), 10.0);
        assert_approx_eq!(bucket1.get(1).unwrap(), 14.0);
        assert_approx_eq!(bucket2.get(0).unwrap(), 4.0);
        assert_approx_eq!(bucket2.get(1).unwrap(), 8.0);
    }
}
