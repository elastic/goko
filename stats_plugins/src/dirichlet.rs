//! # Dirichlet probability
//!
//! We know that the users are quering based on what they want to know about.
//! This has some geometric structure, especially for attackers. We have some
//! prior knowlege about the queries, they should function similarly to the training set.
//! Sections of data that are highly populated should have a higher likelyhood of being
//! queried.
//!
//! This plugin lets us simulate the unknown distribution of the queries of a user in a
//! bayesian way. There may be more applications of this idea, but defending against
//! attackers has been proven.

use rand::prelude::*;
use statrs::function::gamma::{digamma, ln_gamma};

use rand::distributions::{Distribution, Uniform};

use crate::categorical::Categorical;
use crate::parameter_store::DiscreteParams;
use crate::stats_consts::{LN_GAMMA_1024, DIGAMMA_1024};

fn cached_ln_gamma(x: f64) -> f64 {
    if x.fract() == 0.0 && x < 1023.0 {
        LN_GAMMA_1024[x as usize]
    } else {
        ln_gamma(x)
    }
}

fn cached_digamma(x: f64) -> f64 {
    if x.fract() == 0.0 && x < 1023.0 {
        DIGAMMA_1024[x as usize]
    } else {
        digamma(x)
    }
}

/// Simple probability density function for where things go by count
///
#[derive(Debug, Clone, Default)]
pub struct Dirichlet {
    pub(crate) params: DiscreteParams,
}

#[derive(Debug, Clone, Default)]
pub struct DiscreteObservations {
    pub(crate) params: DiscreteParams,
}

impl From<Categorical> for Dirichlet {
    fn from(item: Categorical) -> Self {
        Dirichlet {
            params: item.params,
        }
    }
}

impl Dirichlet {
    /// New all 0 Dirichlet distribution. The child counts are uninitialized
    pub fn new() -> Dirichlet {
        Dirichlet {
            params: DiscreteParams::new(),
        }
    }
    /// Multiplies all parameters by this weight
    pub fn weight(&mut self, weight: f64) {
        self.params.weight(weight);
    }
    /// The total of the parameters. This is a proxy for the total count, and the "concentration" of the distribution
    pub fn total(&self) -> f64 {
        self.params.iter().map(|(_, c)| c).fold(0.0, |x, a| x + a)
    }

    pub fn ln_pdf(&self, categorical: &Categorical) -> Option<f64> {
        if (categorical.params.total() - 1.0).abs() < 0.0001 {
            if self.params.len() != categorical.params.len() || self.params.total() < 0.000000001 {
                return None;
            }

            let mut result = ln_gamma(self.params.total()) + (categorical.params.len() as f64) * categorical.params.total().ln();
            for ((ca, ca_count), (other_ca, other_ca_count)) in
                self.params.iter().zip(categorical.params.iter())
            {
                if ca != other_ca || ca_count <= 0.0 {
                    return None;
                }
                result += ln_gamma(ca_count) + (ca_count - 1.0) * other_ca_count.ln();
            }
            Some(result)
        } else {
            
            if self.params.len() != categorical.params.len() || self.params.total() < 0.000000001 {
                return None;
            }
            let mut result = ln_gamma(self.params.total()) + ln_gamma(categorical.params.total() + 1.0) - ln_gamma(categorical.params.total() + self.params.total());
            for ((ca, ca_count), (other_ca, other_ca_count)) in
                self.params.iter().zip(categorical.params.iter())
            {
                if ca != other_ca || ca_count <= 0.0 {
                    return None;
                }
                if other_ca_count > 0.0 {
                    result += ln_gamma(ca_count + other_ca_count) - ln_gamma(ca_count) - ln_gamma(other_ca_count);
                }
            }
            Some(result)
        }
    }

    /// Gives the probability vector for this
    pub fn param_vec(&self) -> Option<Vec<(u64, f64)>> {
        let total = self.total();
        if total > 0.0 {
            let v: Vec<(u64, f64)> = self.params.iter().map(|(na, f)| (na, f / total)).collect();
            Some(v)
        } else {
            None
        }
    }

    /// Gives the probability vector for this
    pub fn ln_param_vec(&self) -> Option<Vec<(u64, f64)>> {
        let total_ln = self.total().ln();
        if total_ln > 1.0 {
            let v: Vec<(u64, f64)> = self
                .params
                .iter()
                .map(|(na, f)| (na, f.ln() - total_ln))
                .collect();
            Some(v)
        } else {
            None
        }
    }

    fn add_pop(&mut self, loc: u64, count: f64) -> f64 {
        self.params.add_pop(loc, count)
    }

    fn remove_pop(&mut self, loc: u64, count: f64) -> f64 {
        self.params.remove_pop(loc, count)
    }

    /// Adds a single observation to the Dirichlet distribution.
    /// Mutates the distribution in place to the posterior given the new evidence.
    pub fn add_observation(&mut self, loc: u64) {
        self.params.add_pop(loc, 1.0);
    }

    /// Adds a a group of observations to the Dirichlet distribution.
    /// Mutates the distribution in place to the posterior given the new evidence.
    pub fn add_evidence(&mut self, other: &Categorical) {
        for (na, c) in other.params.iter() {
            self.add_pop(na, c);
        }
    }

    /// Computes KL(prior || posterior), where the prior is the distribution
    /// and the posterior is based on the evidence provided.
    pub fn posterior_kl_divergence(&self, other: &Categorical) -> Option<f64> {
        let my_total = self.total();
        let other_total = other.total() + my_total;
        let mut my_total_lng = 0.0;
        let mut other_total_lng = 0.0;
        let mut digamma_portion = 0.0;
        for (other_ca, other_ca_count) in other.params.iter() {
            let ca_count = self.params.get(other_ca).unwrap_or(0.0);
            if ca_count > 0.0 {
                my_total_lng += cached_ln_gamma(ca_count);
                other_total_lng += cached_ln_gamma(other_ca_count + ca_count);
                digamma_portion -= other_ca_count * (cached_digamma(ca_count) - cached_digamma(my_total));
            } else {
                return None;
            }
        }

        let kld = cached_ln_gamma(my_total) - my_total_lng - cached_ln_gamma(other_total)
            + other_total_lng
            + digamma_portion;
        // for floating point errors, sometimes this is -0.000000001
        if kld < 0.0 {
            Some(0.0)
        } else {
            Some(kld)
        }
    }

    pub fn marginal_aic(&self, data: &Categorical) -> Option<f64> {
        self.ln_pdf(data).map(|a| 2.0*self.params.len() as f64 - a)
    }

    /// Samples from the expected PDF of the Dirichlet distribution
    pub fn sample<R: Rng>(&self, rng: &mut R) -> Option<u64> {
        let sum = self.total() as usize;
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

    /// from <http://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/>
    /// We assume that the Dirichlet distribution passed into this one is conditioned on this one! It assumes they have the same keys!
    pub fn supported_kl_divergence(&self, other: &Dirichlet) -> Option<f64> {
        let my_total = self.total();
        if self.params.len() != other.params.len() || my_total < 0.000000001 {
            return None;
        }
        let mut other_total = 0.0;
        let mut my_total_lng = 0.0;
        let mut other_total_lng = 0.0;
        let mut digamma_portion = 0.0;
        for ((ca, ca_count), (other_ca, other_ca_count)) in
            self.params.iter().zip(other.params.iter())
        {
            if ca != other_ca || ca_count <= 0.0 {
                return None;
            }
            other_total += other_ca_count;
            my_total_lng += cached_ln_gamma(ca_count);
            other_total_lng += cached_ln_gamma(other_ca_count);
            digamma_portion +=
                (ca_count - other_ca_count) * (cached_digamma(ca_count) - cached_digamma(my_total));
        }

        let kld = cached_ln_gamma(my_total) - my_total_lng - cached_ln_gamma(other_total)
            + other_total_lng
            + digamma_portion;
        // for floating point errors, sometimes this is -0.000000001
        if kld < 0.0 {
            Some(0.0)
        } else {
            Some(kld)
        }
    }

    /// from <http://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/>
    /// We assume that the Dirichlet distribution passed into this one is conditioned on this one! It assumes they have the same keys!
    pub fn kl_divergence(&self, other: &Dirichlet) -> Option<f64> {
        let my_total = self.total();
        if self.params.len() != other.params.len() || my_total < 0.000000001 {
            return None;
        }
        let mut other_total = 0.0;
        let mut my_total_lng = 0.0;
        let mut other_total_lng = 0.0;
        let mut digamma_portion = 0.0;
        for ((ca, ca_count), (other_ca, other_ca_count)) in self.params.double_iter(&other.params) {
            if ca_count <= 0.0 && other_ca_count > 0.0 {
                return None;
            }
            other_total += other_ca_count;
            my_total_lng += cached_ln_gamma(ca_count);
            other_total_lng += cached_ln_gamma(other_ca_count);
            digamma_portion +=
                (ca_count - other_ca_count) * (cached_digamma(ca_count) - cached_digamma(my_total));
        }

        let kld = cached_ln_gamma(my_total) - my_total_lng - cached_ln_gamma(other_total)
            + other_total_lng
            + digamma_portion;
        // for floating point errors, sometimes this is -0.000000001
        if kld < 0.0 {
            Some(0.0)
        } else {
            Some(kld)
        }
    }

    pub fn tracker(&self) -> DirichletTracker {
        DirichletTracker {
            prior_params: self.clone(),
            evidence_params: Categorical::new(),
            kldiv: 0.0,
            digamma_total: digamma(self.total())
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct DirichletTracker {
    pub(crate) prior_params: Dirichlet,
    pub(crate) evidence_params: Categorical,
    pub(crate) kldiv: f64,
    pub(crate) digamma_total: f64,
}

impl DirichletTracker {
    /// Adds a single observation to the Dirichlet distribution.
    /// Mutates the distribution in place to the posterior given the new evidence.
    pub fn add_observation(&mut self, loc: u64) -> f64 {
        let old_count = self.evidence_params.get(loc);
        let old_total = self.evidence_params.total();
        let new_count = self.evidence_params.add_pop(loc, 1.0);
        let new_total = self.evidence_params.total();
        self.update_kl_div(loc, old_count, old_total, new_count, new_total)
    }

    pub fn remove_observation(&mut self, loc: u64) -> f64 {
        let old_count = self.evidence_params.get(loc);
        let old_total = self.evidence_params.total();
        let new_count = self.evidence_params.remove_pop(loc, 1.0);
        let new_total = self.evidence_params.total();
        assert!(
            old_count.is_some() || old_count == Some(0.0),
            "Can't remove evidence if we've got a 0.0"
        );
        self.update_kl_div(loc, old_count, old_total, new_count, new_total)
    }

    pub fn marginal_aic(&self) -> Option<f64> {
        self.prior_params.marginal_aic(&self.evidence_params)
    }

    fn update_kl_div(
        &mut self,
        loc: u64,
        old_count: Option<f64>,
        old_total: f64,
        new_count: f64,
        new_total: f64,
    ) -> f64 {
        let prior_count = self.prior_params.params.get(loc).expect("Not in the prior!");
        let prior_total = self.prior_params.params.total();
        let diff = if let Some(old_count) = old_count {
            cached_ln_gamma(prior_count + new_count)
                - cached_ln_gamma(prior_count + old_count)
                - (new_count - old_count) * (cached_digamma(prior_count) - self.digamma_total)
        } else {
            cached_ln_gamma(prior_count + new_count)
                - cached_ln_gamma(prior_count)
                - (new_count) * (cached_digamma(prior_count) - self.digamma_total)
        };
        self.kldiv += diff - cached_ln_gamma(prior_total + new_total) + cached_ln_gamma(prior_total + old_total);
        self.kldiv
    }

    fn kl_divergence(&self) -> f64 {
        self.kldiv
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    //use crate::covertree::tests::build_basic_tree;

    #[test]
    fn dirichlet_sanity_test() {
        let mut buckets = Dirichlet::new();
        buckets.add_pop(0, 5.0);
        println!("{:?}", buckets);
        assert_approx_eq!(buckets.params.get(0).unwrap(), 5.0);
        assert_approx_eq!(buckets.supported_kl_divergence(&buckets).unwrap(), 0.0);
    }

    #[test]
    fn dirichlet_mixed_sanity_test() {
        let mut buckets = Dirichlet::new();
        buckets.add_pop(0, 5.0);
        buckets.add_pop(1, 5.0);
        println!("{:?}", buckets);
        assert_approx_eq!(buckets.params.get(0).unwrap(), 5.0);
        assert_approx_eq!(buckets.params.get(1).unwrap(), 5.0);
        assert_approx_eq!(buckets.supported_kl_divergence(&buckets).unwrap(), 0.0);
    }

    #[test]
    fn dirichlet_posterior_sanity_test() {
        let mut buckets = Dirichlet::new();
        buckets.add_pop(0, 3.0);
        buckets.add_pop(1, 5.0);

        let mut categorical = Categorical::new();
        categorical.add_pop(0, 2.0);

        let mut buckets_posterior = buckets.clone();
        buckets_posterior.add_evidence(&categorical);

        println!("Buckets: {:?}", buckets);
        println!("Buckets Posterior: {:?}", buckets_posterior);
        println!("Evidence: {:?}", categorical);
        assert_approx_eq!(buckets_posterior.params.get(0).unwrap(), 5.0);
        assert_approx_eq!(buckets_posterior.params.get(1).unwrap(), 5.0);
        assert_approx_eq!(
            buckets.supported_kl_divergence(&buckets_posterior).unwrap(),
            buckets.posterior_kl_divergence(&categorical).unwrap()
        );
    }

    #[test]
    fn dirichlet_posterior_sanity_test_2() {
        let mut buckets = Dirichlet::new();
        buckets.add_pop(0, 3.0);
        buckets.add_pop(1, 2.0);

        let mut categorical = Categorical::new();
        categorical.add_pop(0, 3.0);
        categorical.add_pop(1, 2.0);

        let mut buckets_posterior = buckets.clone();
        buckets_posterior.add_evidence(&categorical);

        println!("Buckets: {:?}", buckets);
        println!("Buckets Posterior: {:?}", buckets_posterior);
        println!("Evidence: {:?}", categorical);
        assert_approx_eq!(
            buckets.supported_kl_divergence(&buckets_posterior).unwrap(),
            buckets.posterior_kl_divergence(&categorical).unwrap()
        );
        assert_approx_eq!(
            buckets.supported_kl_divergence(&buckets_posterior).unwrap(),
            buckets.kl_divergence(&buckets_posterior).unwrap()
        );
        assert_approx_eq!(
            buckets.supported_kl_divergence(&buckets_posterior).unwrap(),
            0.1789970483832892
        );
    }

    #[test]
    fn dirichlet_tracker_sanity_test() {
        let mut buckets = Dirichlet::new();
        buckets.add_pop(0, 3.0);
        buckets.add_pop(1, 2.0);

        let mut tracker = buckets.tracker();

        let mut buckets_posterior = buckets.clone();
        let posterior_kl_divergence = buckets.supported_kl_divergence(&buckets_posterior).unwrap();
        let tracker_kl_divergence = tracker.kl_divergence();
        println!("{:?}", tracker);
        assert_approx_eq!(posterior_kl_divergence, tracker_kl_divergence);
        for _i in 0..3 {
            buckets_posterior.add_observation(0);
            let posterior_kl_divergence =
                buckets.supported_kl_divergence(&buckets_posterior).unwrap();
            let tracker_kl_divergence = tracker.add_observation(0);
            println!("{:?}", tracker);
            println!("{:?}", buckets_posterior);
            assert_approx_eq!(posterior_kl_divergence, tracker_kl_divergence);
        }
        for _i in 0..2 {
            buckets_posterior.add_observation(1);
            let posterior_kl_divergence =
                buckets.supported_kl_divergence(&buckets_posterior).unwrap();
            let tracker_kl_divergence = tracker.add_observation(1);
            assert_approx_eq!(posterior_kl_divergence, tracker_kl_divergence);
        }
        assert_approx_eq!(tracker.kl_divergence(), 0.1789970483832892);

        for _i in 0..3 {
            buckets_posterior.params.remove_pop(0, 1.0);
            let posterior_kl_divergence =
                buckets.supported_kl_divergence(&buckets_posterior).unwrap();
            let tracker_kl_divergence = tracker.remove_observation(0);
            println!("{:?}", tracker);
            println!("{:?}", buckets_posterior);
            assert_approx_eq!(posterior_kl_divergence, tracker_kl_divergence);
        }
        for _i in 0..2 {
            buckets_posterior.params.remove_pop(1, 1.0);
            let posterior_kl_divergence =
                buckets.supported_kl_divergence(&buckets_posterior).unwrap();
            let tracker_kl_divergence = tracker.remove_observation(1);
            assert_approx_eq!(posterior_kl_divergence, tracker_kl_divergence);
        }
    }

    #[test]
    fn dirichlet_posterior_sanity_test_3() {
        let mut buckets = Dirichlet::new();
        buckets.add_pop(0, 3.0);

        let mut categorical = Categorical::new();
        categorical.add_pop(0, 3.0);

        let mut buckets_posterior = buckets.clone();
        buckets_posterior.add_evidence(&categorical);

        println!("Buckets: {:?}", buckets);
        println!("Buckets Posterior: {:?}", buckets_posterior);
        println!("Evidence: {:?}", categorical);
        assert_approx_eq!(
            buckets.supported_kl_divergence(&buckets_posterior).unwrap(),
            buckets.posterior_kl_divergence(&categorical).unwrap()
        );
        assert_approx_eq!(
            buckets.supported_kl_divergence(&buckets_posterior).unwrap(),
            0.0
        );
    }

    #[test]
    fn dirichlet_kl_sanity_test() {
        let mut bucket1 = Dirichlet::new();
        bucket1.add_pop(0, 6.0);
        bucket1.add_pop(1, 6.0);
        println!("{:?}", bucket1);

        let mut bucket2 = Dirichlet::new();
        bucket2.add_pop(0, 3.0);
        bucket2.add_pop(1, 9.0);
        println!("{:?}", bucket2);

        let mut bucket3 = Dirichlet::new();
        bucket3.add_pop(0, 5.5);
        bucket3.add_pop(1, 6.5);
        println!("{:?}", bucket3);
        println!(
            "{:?}, {}",
            bucket1.supported_kl_divergence(&bucket2).unwrap(),
            bucket1.supported_kl_divergence(&bucket3).unwrap()
        );
        assert!(
            bucket1.supported_kl_divergence(&bucket2).unwrap()
                > bucket1.supported_kl_divergence(&bucket3).unwrap()
        );
    }

    #[test]
    fn dirichlet_aic_sanity_test() {
        let mut posterior = Dirichlet::new();
        posterior.add_pop(0, 2.0);
        posterior.add_pop(1, 2.0);
        println!("{:?}", posterior);

        let mut data1 = Categorical::new();
        data1.add_pop(0, 2.0);
        data1.add_pop(1, 0.0);
        println!("{:?}", data1);

        let mut data2 = Categorical::new();
        data2.add_pop(0, 2.0);
        data2.add_pop(1, 2.0);
        println!("{:?}", data2);
        println!(
            "{:?}, {}",
            posterior.marginal_aic(&data1).unwrap(),
            posterior.marginal_aic(&data2).unwrap()
        );
        assert!(
            posterior.marginal_aic(&data1).unwrap()
                > posterior.marginal_aic(&data2).unwrap()
        );
    }

    /*
    #[test]
    fn dirichlet_tree_probs_test() {
        let mut tree = build_basic_tree();
        tree.add_plugin::<GokoDirichlet>(GokoDirichlet::default());
        assert!(tree.reader().get_plugin_and::<GokoDirichlet,_,_>(|p| {
            // Checked these probs by hand
            assert_approx_eq!(p.cond_ln_probs.get(&( -2,2)).unwrap(),-0.5108256237659905);
            true
        }).unwrap());
    }
    */
}
