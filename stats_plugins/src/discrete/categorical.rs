//! # Categorical Distribution
//!
//! Simple probability distribution that enables you to simulated the rough
//! distribution of data in the tree.
use super::parameter_store::DiscreteParams;

use rand::distributions::{Distribution, Uniform};
use rand::Rng;

use super::data::DiscreteData;

/// Simple probability density function for where things go by count
/// Stored as a flat vector in the order of the node addresses.
#[derive(Debug, Clone, Default)]
pub struct Categorical {
    pub(crate) params: DiscreteParams,
}

impl From<DiscreteData> for Categorical {
    fn from(item: DiscreteData) -> Self {
        Categorical {
            params: item.params,
        }
    }
}

impl Categorical {
    /// Creates a new empty bucket probability
    pub fn new() -> Categorical {
        Categorical {
            params: DiscreteParams::new(),
        }
    }

    /// Gives the probability vector for this
    pub fn prob_vector(&self) -> Option<Vec<(u64, f64)>> {
        let total = self.params.total();
        if total > 0.0 {
            let v: Vec<(u64, f64)> = self.params.iter().map(|(na, f)| (na, f / total)).collect();
            Some(v)
        } else {
            None
        }
    }

    pub fn get(&self, loc: u64) -> Option<f64> {
        self.params.get(loc)
    }

    /// Pass none if you want to test for a singleton, returns 0 if
    pub fn ln_pdf(&self, loc: u64) -> Option<f64> {
        let total = self.params.total();
        if total > 0.0 {
            self.params
                .get(loc)
                .map(|a| a.ln() - total.ln())
                .or(Some(std::f64::NEG_INFINITY))
        } else {
            None
        }
    }

    /// Samples from the given categorical distribution
    pub fn sample<R: Rng>(&self, rng: &mut R) -> Option<u64> {
        let sum = self.params.total() as usize;
        let uniform = Uniform::from(0..sum);
        let sample = uniform.sample(rng) as f64;

        let mut count = 0.0;
        for (a, c) in self.params.iter() {
            count += c;
            if sample < count {
                return Some(a);
            }
        }
        None
    }

    /// Computes the KL divergence of two bucket probs.
    /// KL(self || other)
    /// Returns None if the support of the self is not a subset of the support of the other
    ///
    /// This assumes that the support of this distribution is a superset of the other distribution.
    /// Returns None if the supports don't match, or it's undefined.
    pub fn supported_kl_divergence(&self, other: &Categorical) -> Option<f64> {
        let my_total = self.params.total();
        let other_total = other.params.total();
        if (my_total == 0.0 || other_total == 0.0) || (self.params.len() != other.params.len()) {
            None
        } else {
            let ln_total = my_total.ln() - other_total.ln();
            let mut sum: f64 = 0.0;
            for ((ca, ca_count), (other_ca, other_ca_count)) in
                self.params.iter().zip(other.params.iter())
            {
                if ca_count <= 0.0 || ca != other_ca {
                    return None;
                } else {
                    sum += (ca_count / my_total) * (ca_count.ln() - other_ca_count.ln() - ln_total);
                }
            }
            Some(sum)
        }
    }

    /// Computes the KL divergence of two bucket probs.
    /// KL(self || other)
    /// Returns None if the support of the self is not a subset of the support of the other
    ///
    /// This assumes that the support of this distribution is a superset of the other distribution.
    /// Returns None if it's undefined.
    pub fn kl_divergence(&self, other: &Categorical) -> Option<f64> {
        let my_total = self.params.total();
        let other_total = other.params.total();
        if my_total == 0.0 || other_total == 0.0 {
            None
        } else {
            let ln_total = my_total.ln() - other_total.ln();
            let mut sum: f64 = 0.0;
            for ((ca, ca_count), (other_ca, other_ca_count)) in
                self.params.double_iter(&other.params)
            {
                if (ca_count <= 0.0 && other_ca_count > 0.0) || ca != other_ca  {
                    return None;
                }
                sum += (ca_count / my_total) * (ca_count.ln() - other_ca_count.ln() - ln_total);
            }
            Some(sum)
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    //use crate::tree::tests::build_basic_tree;

    #[test]
    fn empty_bucket_sanity_test() {
        let buckets = Categorical::new();
        assert_eq!(buckets.ln_pdf(0), None);
        assert_eq!(buckets.ln_pdf(1), None);
        assert_eq!(buckets.supported_kl_divergence(&buckets), None)
    }

    #[test]
    fn bucket_sanity_test() {
        let mut buckets = Categorical::new();
        buckets.params.add_pop(0, 5.0);
        assert_approx_eq!(buckets.ln_pdf(0).unwrap(), 0.0);
        assert_approx_eq!(buckets.supported_kl_divergence(&buckets).unwrap(), 0.0);
        assert_eq!(buckets.ln_pdf(1), Some(std::f64::NEG_INFINITY));
    }

    #[test]
    fn mixed_bucket_sanity_test() {
        let mut bucket1 = Categorical::new();
        bucket1.params.add_pop(0, 6.0);
        bucket1.params.add_pop(1, 6.0);
        println!("{:?}", bucket1);

        let mut bucket2 = Categorical::new();
        bucket2.params.add_pop(0, 4.0);
        bucket2.params.add_pop(1, 8.0);
        println!("{:?}", bucket2);

        assert_approx_eq!(bucket1.ln_pdf(0).unwrap(), (0.5f64).ln());
        assert_approx_eq!(bucket2.ln_pdf(1).unwrap(), (0.666666666f64).ln());
        assert_approx_eq!(bucket1.supported_kl_divergence(&bucket1).unwrap(), 0.0);
        assert_approx_eq!(
            bucket1.supported_kl_divergence(&bucket2).unwrap(),
            0.05889151782
        );
        assert_approx_eq!(
            bucket2.supported_kl_divergence(&bucket1).unwrap(),
            0.05663301226
        );

        assert_approx_eq!(bucket1.kl_divergence(&bucket1).unwrap(), 0.0);
        assert_approx_eq!(bucket1.kl_divergence(&bucket2).unwrap(), 0.05889151782);
        assert_approx_eq!(bucket2.kl_divergence(&bucket1).unwrap(), 0.05663301226);
    }

    #[test]
    fn supported_kldiv_handles_zero_probs() {
        let mut bucket1 = Categorical::new();
        bucket1.params.add_pop(0, 6.0);
        bucket1.params.add_pop(1, 6.0);
        println!("{:?}", bucket1);

        let mut bucket2 = Categorical::new();
        bucket2.params.add_pop(0, 4.0);
        assert_eq!(bucket1.supported_kl_divergence(&bucket2), None);
        assert_eq!(bucket2.supported_kl_divergence(&bucket1), None);

        let mut bucket3 = Categorical::new();
        bucket3.params.add_pop(0, 4.0);
        bucket3.params.add_pop(1, 0.0);
        println!("{:?}", bucket3);
        assert_eq!(
            bucket1.supported_kl_divergence(&bucket3),
            Some(std::f64::INFINITY)
        );
        assert_eq!(bucket3.supported_kl_divergence(&bucket1), None);

        let mut bucket4 = Categorical::new();
        bucket4.params.add_pop(0, 0.0);
        bucket4.params.add_pop(2, 4.0);
        assert_eq!(bucket1.supported_kl_divergence(&bucket4), None);
        assert_eq!(bucket4.supported_kl_divergence(&bucket1), None);
    }

    #[test]
    fn kldiv_handles_zero_probs() {
        let mut bucket1 = Categorical::new();
        bucket1.params.add_pop(0, 6.0);
        bucket1.params.add_pop(1, 6.0);
        println!("{:?}", bucket1);

        let mut bucket2 = Categorical::new();
        bucket2.params.add_pop(0, 4.0);
        assert_eq!(bucket1.kl_divergence(&bucket2), Some(std::f64::INFINITY));
        assert_eq!(bucket2.kl_divergence(&bucket1), None);

        let mut bucket3 = Categorical::new();
        bucket3.params.add_pop(0, 4.0);
        bucket3.params.add_pop(1, 0.0);
        println!("{:?}", bucket3);
        assert_eq!(bucket1.kl_divergence(&bucket3), Some(std::f64::INFINITY));
        assert_eq!(bucket3.kl_divergence(&bucket1), None);

        let mut bucket4 = Categorical::new();
        bucket4.params.add_pop(0, 4.0);
        bucket4.params.add_pop(2, 4.0);
        assert_eq!(bucket1.kl_divergence(&bucket4), None);
        assert_eq!(bucket4.kl_divergence(&bucket1), None);
    }
}
