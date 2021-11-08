//! # DiscreteData Distribution
//!
//! Simple probability distribution that enables you to simulated the rough
//! distribution of data in the tree.
use super::parameter_store::DiscreteParams;
use core_goko::*;

/// Simple probability density function for where things go by count
/// Stored as a flat vector in the order of the node addresses.
#[derive(Debug, Clone, Default)]
pub struct DiscreteData {
    pub(crate) params: DiscreteParams,
}

impl From<&[(u64, f64)]> for DiscreteData {
    fn from(vals: &[(u64, f64)]) -> DiscreteData {
        let params = DiscreteParams::from(vals);
        DiscreteData { params }
    }
}

impl From<&[(NodeAddress, f64)]> for DiscreteData {
    fn from(vals: &[(NodeAddress, f64)]) -> DiscreteData {
        let params = DiscreteParams::from(vals);
        DiscreteData { params }
    }
}

impl DiscreteData {
    /// Creates a new empty bucket probability
    pub fn new() -> DiscreteData {
        DiscreteData {
            params: DiscreteParams::new(),
        }
    }

    /// Gives the probability vector for this
    pub fn data_vec(&self) -> Vec<(NodeAddress, f64)> {
        self.params.iter().collect()
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

    pub fn add_pop(&mut self, loc: NodeAddress, count: f64) -> (f64, f64) {
        self.params.add_pop(loc, count)
    }

    pub fn remove_pop(&mut self, loc: NodeAddress, count: f64) -> (f64, f64) {
        self.params.remove_pop(loc, count)
    }

    pub fn get(&self, loc: NodeAddress) -> Option<f64> {
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
        assert_eq!(buckets.get(0.into()), None);
        assert_eq!(buckets.total(), 0.0);
        buckets.add_pop(0.into(), 5.0);
        assert_eq!(buckets.get(0.into()), Some(5.0));
        assert_eq!(buckets.total(), 5.0);
    }

    #[test]
    fn merge_test() {
        let mut bucket1 = DiscreteData::from(&[(0u64, 6.0f64), (1, 6.0)][..]);
        println!("{:?}", bucket1);
        let bucket2 = DiscreteData::from(&[(0u64, 4.0f64), (1, 8.0)][..]);
        println!("{:?}", bucket2);
        bucket1.merge(&bucket2);
        assert_approx_eq!(bucket1.get(0.into()).unwrap(), 10.0);
        assert_approx_eq!(bucket1.get(1.into()).unwrap(), 14.0);
        assert_approx_eq!(bucket2.get(0.into()).unwrap(), 4.0);
        assert_approx_eq!(bucket2.get(1.into()).unwrap(), 8.0);
    }
}
