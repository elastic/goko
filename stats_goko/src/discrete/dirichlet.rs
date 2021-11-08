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
use core_goko::*;
use serde::{Deserialize, Serialize};
use rand::prelude::*;
use statrs::function::gamma::{digamma, ln_gamma};
use rand::distributions::{Distribution, Uniform};

use super::categorical::Categorical;
use super::data::DiscreteData;
use super::parameter_store::{
    DiscreteParams, DiscreteParamsDoubleIter, DiscreteParamsIndexes, DiscreteParamsIter,
};
use super::stats_consts::{DIGAMMA_1024, LN_GAMMA_1024};

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
#[derive(Debug, Clone, Default, Deserialize, Serialize)]
pub struct Dirichlet {
    indexes: DiscreteParamsIndexes,
    values: Vec<f64>,
    total: f64,
}

impl From<DiscreteData> for Dirichlet {
    fn from(item: DiscreteData) -> Self {
        Dirichlet {
            indexes: item.params.indexes,
            values: item.params.values,
            total: item.params.total,
        }
    }
}

impl From<&[(u64, f64)]> for Dirichlet {
    fn from(vals: &[(u64, f64)]) -> Dirichlet {
        let params = DiscreteParams::from(vals);
        Dirichlet {
            indexes: params.indexes,
            values: params.values,
            total: params.total,
        }
    }
}

impl From<&[(NodeAddress, f64)]> for Dirichlet {
    fn from(vals: &[(NodeAddress, f64)]) -> Dirichlet {
        let params = DiscreteParams::from(vals);
        Dirichlet {
            indexes: params.indexes,
            values: params.values,
            total: params.total,
        }
    }
}

impl Dirichlet {
    /// New all 0 Dirichlet distribution. The child counts are uninitialized
    pub fn new() -> Dirichlet {
        Dirichlet {
            indexes: DiscreteParamsIndexes::new(),
            values: Vec::new(),
            total: 0.0,
        }
    }
    /// Multiplies all parameters by this weight
    pub fn weight(&mut self, weight: f64) {
        self.values.iter_mut().for_each(|p| *p *= weight);
    }
    /// The total of the parameters. This is a proxy for the total count, and the "concentration" of the distribution
    pub fn total(&self) -> f64 {
        self.total
    }

    pub fn get_alpha(&self, loc: NodeAddress) -> Option<f64> {
        self.indexes.get_from(loc, &self.values)
    }

    pub fn ln_pdf(&self, categorical: &Categorical) -> Option<f64> {
        if self.indexes != categorical.params.indexes || self.total < 0.000000001 {
            return None;
        }
        let mut result = cached_ln_gamma(self.total);
        for (ca_count, other_ca_count) in self.values.iter().zip(categorical.params.values.iter()) {
            result += cached_ln_gamma(*ca_count) + (*ca_count - 1.0) * other_ca_count.ln();
        }
        Some(result)
    }

    pub fn ln_likelihood(&self, data: &DiscreteData) -> Option<f64> {
        if self.indexes != data.params.indexes || self.total < 0.000000001 {
            return None;
        }
        let mut result = cached_ln_gamma(self.total) + cached_ln_gamma(data.total() + 1.0)
            - cached_ln_gamma(data.total() + self.total());
        for (ca_count, other_ca_count) in self.values.iter().zip(data.params.values.iter()) {
            if *other_ca_count > 0.0 {
                result += cached_ln_gamma(*ca_count + *other_ca_count)
                    - cached_ln_gamma(*ca_count)
                    - cached_ln_gamma(*other_ca_count + 1.0);
            }
        }
        Some(result)
    }

    pub fn maximum_ln_likelihood_of_k_items(&self, k: f64) -> f64 {
        let mut result = cached_ln_gamma(self.total) + cached_ln_gamma(k + 1.0)
            - cached_ln_gamma(k + self.total());
        for ca_count in self.values.iter() {
            let other_ca_count = k*(*ca_count)/self.total();
            result += cached_ln_gamma(*ca_count + other_ca_count)
                - cached_ln_gamma(*ca_count)
                - cached_ln_gamma(other_ca_count + 1.0);
        }
        result
    }

    /// Gives the probability vector for this
    pub fn param_vec(&self) -> Vec<(NodeAddress, f64)> {
        self.iter().collect()
    }

    /// Adds a single observation to the Dirichlet distribution.
    /// Mutates the distribution in place to the posterior given the new evidence.
    pub fn add_observation(&mut self, loc: NodeAddress) {
        self.total += 1.0;
        self.indexes.add(loc, 1.0, &mut self.values);
    }

    /// Adds a a group of observations to the Dirichlet distribution.
    /// Mutates the distribution in place to the posterior given the new evidence.
    pub fn add_observations(&mut self, other: &DiscreteData) {
        self.total += other.total();
        for (na, c) in other.params.iter() {
            self.indexes.add(na, c, &mut self.values);
        }
    }

    /// Computes KL(prior || posterior), where the prior is the distribution
    /// and the posterior is based on the evidence provided.
    pub fn posterior_kl_div(&self, other: &DiscreteData) -> Option<f64> {
        let my_total = self.total();
        let other_total = other.total() + my_total;
        let mut my_total_lng = 0.0;
        let mut other_total_lng = 0.0;
        let mut digamma_portion = 0.0;
        for (other_ca, other_ca_count) in other.params.iter() {
            let ca_count = self.indexes.get_from(other_ca, &self.values).unwrap_or(0.0);
            if ca_count > 0.0 {
                my_total_lng += cached_ln_gamma(ca_count);
                other_total_lng += cached_ln_gamma(other_ca_count + ca_count);
                digamma_portion -=
                    other_ca_count * (cached_digamma(ca_count) - cached_digamma(my_total));
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

    pub fn marginal_aic(&self, data: &DiscreteData) -> Option<f64> {
        self.ln_likelihood(data)
            .map(|a| 2.0 * self.values.len() as f64 - a)
    }

    /// Samples from the expected PDF of the Dirichlet distribution
    pub fn marginal_sample<R: Rng>(&self, rng: &mut R) -> Option<NodeAddress> {
        let sum = self.total() as usize;
        let uniform = Uniform::from(0..sum);
        let sample = uniform.sample(rng) as f64;

        let mut count = 0.0;
        for (a, c) in self.iter() {
            count += c;
            if sample < count {
                return Some(a);
            }
        }
        None
    }

    /// from <http://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/>
    /// We assume that the Dirichlet distribution passed into this one is conditioned on this one! It assumes they have the same keys!
    pub fn supported_kl_div(&self, other: &Dirichlet) -> Option<f64> {
        let my_total = self.total();
        if self.indexes != other.indexes || my_total < 0.000000001 {
            return None;
        }
        let mut other_total = 0.0;
        let mut my_total_lng = 0.0;
        let mut other_total_lng = 0.0;
        let mut digamma_portion = 0.0;
        for (ca_count, other_ca_count) in self.values.iter().zip(other.values.iter()) {
            if *ca_count <= 0.0 {
                return None;
            }
            other_total += other_ca_count;
            my_total_lng += cached_ln_gamma(*ca_count);
            other_total_lng += cached_ln_gamma(*other_ca_count);
            digamma_portion += (ca_count - other_ca_count)
                * (cached_digamma(*ca_count) - cached_digamma(my_total));
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
    pub fn kl_div(&self, other: &Dirichlet) -> Option<f64> {
        let my_total = self.total();
        if my_total < 0.000000001 {
            return None;
        }
        let mut other_total = 0.0;
        let mut my_total_lng = 0.0;
        let mut other_total_lng = 0.0;
        let mut digamma_portion = 0.0;
        for ((_ca, ca_count), (_other_ca, other_ca_count)) in self.double_iter(&other) {
            if ca_count <= 0.0 && 0.0 < other_ca_count {
                return None;
            }
            other_total += other_ca_count;
            my_total_lng += cached_ln_gamma(ca_count);
            other_total_lng += cached_ln_gamma(other_ca_count);
            digamma_portion +=
                (ca_count - other_ca_count) * (cached_digamma(ca_count) - cached_digamma(my_total));
        }

        let kld = cached_ln_gamma(my_total) - cached_ln_gamma(other_total) - my_total_lng
            + other_total_lng
            + digamma_portion;
        // for floating point errors, sometimes this is -0.000000001
        if kld < 0.0 {
            Some(0.0)
        } else {
            Some(kld)
        }
    }

    pub fn iter(&self) -> DiscreteParamsIter {
        DiscreteParamsIter {
            index_iter: self.indexes.iter(),
            value_iter: self.values.iter(),
        }
    }

    pub fn double_iter<'b>(&self, other: &'b Dirichlet) -> DiscreteParamsDoubleIter<'_, 'b> {
        let mut iter_a = self.iter();
        let mut iter_b = other.iter();
        let val_a = iter_a.next();
        let val_b = iter_b.next();
        DiscreteParamsDoubleIter {
            iter_a,
            iter_b,
            val_a,
            val_b,
        }
    }

    pub fn tracker(&self) -> DirichletTracker {
        DirichletTracker {
            total_alpha: self.total(),
            sparse_alpha: self.total(),
            observation_total: 0.0,
            indexes: self.indexes.clone(),
            values: self
                .values
                .iter()
                .map(|alpha| DiscreteTrackerEntry::new(*alpha))
                .collect(),
            kl_div: 0.0,
            digamma_total: cached_digamma(self.total()),
            ln_gamma_total: cached_ln_gamma(self.total()),
            num_buckets: self.values.len(),
            mll: 0.0,
            kl_div_const: 0.0,
            mll_const: 0.0,
        }
    }
}

#[derive(Debug, Clone, Default)]
struct DiscreteTrackerEntry {
    alpha: f64,
    observations: f64,
    alpha_digamma: f64,
    alpha_ln_gamma: f64,
    mll: f64,
    kl_div: f64,
}

impl DiscreteTrackerEntry {
    fn new(alpha: f64) -> DiscreteTrackerEntry {
        DiscreteTrackerEntry {
            alpha: alpha,
            observations: 0.0,
            alpha_digamma: digamma(alpha),
            alpha_ln_gamma: ln_gamma(alpha),
            mll: 0.0,
            kl_div: 0.0,
        }
    }

    fn weight(&mut self, weight: f64) {
        assert!(self.observations == 0.0, "Can't reweight after observations");
        self.alpha *= weight;
        self.alpha_digamma = digamma(self.alpha);
        self.alpha_ln_gamma = ln_gamma(self.alpha);
    }

    fn update(&mut self, observation_diff: f64, digamma_total: f64) -> ((f64, f64), (f64, f64)) {
        self.observations += observation_diff;
        debug_assert!(self.observations >= 0.0);
        let old_mll = self.mll;
        let old_kl = self.kl_div;
        if self.observations > 1e-7 {
            let alpha_new_ln_gamma = ln_gamma(self.alpha + self.observations);
            self.mll = alpha_new_ln_gamma - self.alpha_ln_gamma - ln_gamma(self.observations + 1.0);
            self.kl_div = alpha_new_ln_gamma
                - self.alpha_ln_gamma
                - self.observations * (self.alpha_digamma - digamma_total);
        } else {
            self.observations = 0.0;
            self.mll = 0.0;
            self.kl_div = 0.0;
        }
        ((old_mll, self.mll), (old_kl, self.kl_div))
    }
}

#[derive(Debug, Clone, Default)]
pub struct DirichletTracker {
    total_alpha: f64,
    sparse_alpha: f64,
    observation_total: f64,
    indexes: DiscreteParamsIndexes,
    values: Vec<DiscreteTrackerEntry>,
    kl_div: f64,
    kl_div_const: f64,
    digamma_total: f64,
    ln_gamma_total: f64,
    num_buckets: usize,
    mll: f64,
    mll_const: f64,
}


impl DirichletTracker {
    pub fn sparse(total_alpha: f64, num_buckets: usize) -> DirichletTracker {
        DirichletTracker {
            total_alpha,
            sparse_alpha: 0.0,
            observation_total: 0.0,
            indexes: DiscreteParamsIndexes::new(),
            values: Vec::new(),
            kl_div: 0.0,
            kl_div_const: 0.0,
            digamma_total: digamma(total_alpha),
            ln_gamma_total: ln_gamma(total_alpha),
            num_buckets,
            mll_const: 0.0,
            mll: 0.0,
        }
    }

    pub fn set_alpha(&mut self, loc: NodeAddress, alpha: f64) {
        match self.indexes.get_or_insert(loc) {
            Ok(index) => {
                let old_data = &mut self.values[index];
                assert!(
                    0.0 == old_data.alpha || old_data.alpha == alpha,
                    "Cannot update an alpha that was already set"
                );
                self.sparse_alpha += alpha - old_data.alpha;
                old_data.alpha = alpha;
                old_data.alpha_digamma = cached_digamma(alpha);
            }
            Err(index) => {
                self.sparse_alpha += alpha;
                let val = DiscreteTrackerEntry::new(alpha);
                self.values.insert(index, val);
            }
        }
        assert!(
            self.sparse_alpha <= self.total_alpha + 1e-7,
            "The prior's total cannot exceed the predetermined alpha."
        );
    }
    /*
    pub fn prior(&self) -> &Dirichlet {
        &self.prior_params
    }

    pub fn data(&self) -> &DiscreteData {
        &self.observed_data
    }
    */

    pub fn get_observations(&self, loc: NodeAddress) -> Option<f64> {
        self.indexes.get(loc).map(|i| self.values[i].observations)
    }

    pub fn get_alpha(&self, loc: NodeAddress) -> Option<f64> {
        self.indexes.get(loc).map(|i| self.values[i].alpha)
    }

    pub fn marginal_posterior_probs(&self) -> Vec<(NodeAddress, f64)> {
        self.values.iter().zip(self.indexes.iter()).map(|(v,a)| (a.into(),(v.alpha + v.observations)/ (self.total_alpha + self.observation_total))).collect()
    }

    pub fn total_alpha(&self) -> f64 {
        self.total_alpha
    }

    pub fn total_observations(&self) -> f64 {
        self.observation_total
    }

    pub fn weight_alphas(&mut self, weight: f64) {
        assert!(self.observation_total == 0.0, "Can't reweight after observations");
        self.total_alpha *= weight;
        self.values.iter_mut().for_each(|v| v.weight(weight));
    }

    pub fn remove_observation(&mut self, loc: NodeAddress) -> &mut Self {
        self.inc_observation(loc, -1.0)
    }
    pub fn add_observation(&mut self, loc: NodeAddress) -> &mut Self {
        self.inc_observation(loc, 1.0)
    }

    /// Adds a single observation to the Dirichlet distribution.
    /// Mutates the distribution in place to the posterior given the new evidence.
    pub fn inc_observation(&mut self, loc: NodeAddress, change: f64) -> &mut Self {
        let index = self
            .indexes
            .get(loc)
            .expect("Please set the prior observations before adding evidence");
        let ((old_mll_term, new_mll_term), (old_kl_term, new_kl_term)) =
            self.values[index].update(change, self.digamma_total);
        self.observation_total += change;
        let other_total_lng = ln_gamma(self.observation_total + self.total_alpha);
        self.kl_div -= self.kl_div_const + old_kl_term;
        self.mll -= self.mll_const + old_mll_term;
        self.mll_const =
            self.ln_gamma_total + ln_gamma(self.observation_total + 1.0) - other_total_lng;
        self.kl_div_const = self.ln_gamma_total - other_total_lng;
        self.kl_div += self.kl_div_const + new_kl_term;
        self.mll += self.mll_const + new_mll_term;
        self
    }

    pub fn kl_div(&self) -> f64 {
        self.kl_div
    }

    pub fn marginal_aic(&self) -> f64 {
        2.0 * self.num_buckets as f64 - self.mll
    }

    pub fn mll(&self) -> f64 {
        self.mll
    }
    
    pub fn mll_const(&self) -> f64 {
        self.mll_const
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    //use crate::covertree::tests::build_basic_tree;

    #[test]
    fn dirichlet_sanity_test() {
        let mut buckets = Dirichlet::new();
        buckets.add_observation(0.into());
        buckets.add_observation(0.into());
        buckets.add_observation(0.into());
        buckets.add_observation(0.into());
        buckets.add_observation(0.into());
        println!("{:?}", buckets);
        assert_approx_eq!(buckets.get_alpha(0.into()).unwrap(), 5.0);
        assert_approx_eq!(buckets.supported_kl_div(&buckets).unwrap(), 0.0);
    }

    #[test]
    fn dirichlet_mixed_sanity_test() {
        let buckets = Dirichlet::from(&[(0, 5.0), (1, 5.0)][..]);
        println!("{:?}", buckets);
        assert_approx_eq!(buckets.get_alpha(0.into()).unwrap(), 5.0);
        assert_approx_eq!(buckets.get_alpha(1.into()).unwrap(), 5.0);
        assert_approx_eq!(buckets.supported_kl_div(&buckets).unwrap(), 0.0);
    }

    #[test]
    fn dirichlet_posterior_sanity_test() {
        let buckets = Dirichlet::from(&[(0, 3.0), (1, 5.0)][..]);

        let data = DiscreteData::from(&[(0, 2.0)][..]);

        let mut buckets_posterior = buckets.clone();
        buckets_posterior.add_observations(&data);

        println!("Buckets: {:?}", buckets);
        println!("Buckets Posterior: {:?}", buckets_posterior);
        println!("Evidence: {:?}", data);
        assert_approx_eq!(buckets_posterior.get_alpha(0.into()).unwrap(), 5.0);
        assert_approx_eq!(buckets_posterior.get_alpha(1.into()).unwrap(), 5.0);
        assert_approx_eq!(
            buckets.supported_kl_div(&buckets_posterior).unwrap(),
            buckets.posterior_kl_div(&data).unwrap()
        );
    }

    #[test]
    fn dirichlet_posterior_sanity_test_2() {
        let data = DiscreteData::from(&[(0, 3.0), (1, 2.0)][..]);

        let categorical = Categorical::from(data.clone());
        let buckets = Dirichlet::from(data.clone());

        let mut buckets_posterior = buckets.clone();
        buckets_posterior.add_observations(&data);

        println!("Buckets: {:?}", buckets);
        println!("Buckets Posterior: {:?}", buckets_posterior);
        println!("Evidence: {:?}", categorical);
        assert_approx_eq!(
            buckets.supported_kl_div(&buckets_posterior).unwrap(),
            buckets.posterior_kl_div(&data).unwrap()
        );
        assert_approx_eq!(
            buckets.supported_kl_div(&buckets_posterior).unwrap(),
            buckets.kl_div(&buckets_posterior).unwrap()
        );
        assert_approx_eq!(
            buckets.supported_kl_div(&buckets_posterior).unwrap(),
            0.1789970483832892
        );
    }

    #[test]
    fn dirichlet_tracker_sanity_test() {
        let buckets = Dirichlet::from(&[(0, 3.0), (1, 2.0)][..]);

        let mut tracker = buckets.tracker();

        let mut true_data = DiscreteData::new();
        true_data.add_pop(0.into(), 0.0);
        true_data.add_pop(1.into(), 0.0);
        let posterior_kl_div = buckets.posterior_kl_div(&true_data).unwrap();
        let tracker_kl_div = tracker.kl_div();
        println!("{:#?}", tracker);
        assert_approx_eq!(posterior_kl_div, tracker_kl_div);
        for _i in 0..3 {
            println!("================================================");
            true_data.add_pop(0.into(), 1.0);
            let posterior_kl_div = buckets.posterior_kl_div(&true_data).unwrap();
            let tracker_kl_div = tracker.add_observation(0.into()).kl_div();
            let posterior_aic = buckets.marginal_aic(&true_data).unwrap();
            let tracker_aic = tracker.marginal_aic();
            println!("{:#?}", tracker);
            println!("{:#?}", true_data);
            assert_approx_eq!(posterior_kl_div, tracker_kl_div);
            assert_approx_eq!(posterior_aic, tracker_aic);
        }
        for _i in 0..2 {
            println!("================================================");
            true_data.add_pop(1.into(), 1.0);
            let posterior_kl_div = buckets.posterior_kl_div(&true_data).unwrap();
            let tracker_kl_div = tracker.add_observation(1.into()).kl_div();
            let posterior_aic = buckets.marginal_aic(&true_data).unwrap();
            let tracker_aic = tracker.marginal_aic();
            println!("{:#?}", tracker);
            println!("{:#?}", true_data);
            assert_approx_eq!(posterior_kl_div, tracker_kl_div);
            assert_approx_eq!(posterior_aic, tracker_aic);
        }
        assert_approx_eq!(tracker.kl_div(), 0.1789970483832892);

        for _i in 0..3 {
            println!("================================================");
            true_data.remove_pop(0.into(), 1.0);
            let posterior_kl_div = buckets.posterior_kl_div(&true_data).unwrap();
            let tracker_kl_div = tracker.remove_observation(0.into()).kl_div();
            let posterior_aic = buckets.marginal_aic(&true_data).unwrap();
            let tracker_aic = tracker.marginal_aic();
            println!("{:#?}", tracker);
            println!("{:#?}", true_data);
            assert_approx_eq!(posterior_kl_div, tracker_kl_div);
            assert_approx_eq!(posterior_aic, tracker_aic);
        }
        for _i in 0..2 {
            println!("================================================");
            true_data.remove_pop(1.into(), 1.0);
            let posterior_kl_div = buckets.posterior_kl_div(&true_data).unwrap();
            let tracker_kl_div = tracker.remove_observation(1.into()).kl_div();
            let posterior_aic = buckets.marginal_aic(&true_data).unwrap();
            let tracker_aic = tracker.marginal_aic();
            println!("{:#?}", tracker);
            println!("{:#?}", true_data);
            assert_approx_eq!(posterior_kl_div, tracker_kl_div);
            assert_approx_eq!(posterior_aic, tracker_aic);
        }
    }

    #[test]
    fn dirichlet_sparse_tracker_sanity_test() {
        let mut tracker = DirichletTracker::sparse(5.0, 2);
        // The sparse adds one to these
        let buckets = Dirichlet::from(&[(0, 3.0), (1, 2.0)][..]);

        let mut dense_tracker = buckets.tracker();

        tracker.set_alpha(0.into(), 3.0);
        let true_tracker_kl_div = dense_tracker.kl_div();
        let tracker_kl_div = tracker.kl_div();
        println!("{:#?}", tracker);
        println!("{:?}", dense_tracker);
        assert_approx_eq!(true_tracker_kl_div, tracker_kl_div);
        for _i in 0..3 {
            let true_tracker_kl_div = dense_tracker.add_observation(0.into()).kl_div();
            let tracker_kl_div = tracker.add_observation(0.into()).kl_div();
            println!("{:#?}", tracker);
            println!("{:?}", dense_tracker);
            assert_approx_eq!(true_tracker_kl_div, tracker_kl_div);
        }
        tracker.set_alpha(1.into(), 2.0);
        for _i in 0..2 {
            let true_tracker_kl_div = dense_tracker.add_observation(1.into()).kl_div();
            let tracker_kl_div = tracker.add_observation(1.into()).kl_div();
            println!("{:#?}", tracker);
            println!("{:?}", dense_tracker);
            assert_approx_eq!(true_tracker_kl_div, tracker_kl_div);
        }
        assert_approx_eq!(tracker.kl_div(), 0.1789970483832892);
    }

    #[test]
    fn dirichlet_posterior_sanity_test_3() {
        let buckets = Dirichlet::from(&[(0, 3.0)][..]);
        let data = DiscreteData::from(&[(0, 3.0)][..]);

        let mut buckets_posterior = buckets.clone();
        buckets_posterior.add_observations(&data);

        println!("Buckets: {:?}", buckets);
        println!("Buckets Posterior: {:?}", buckets_posterior);
        println!("Evidence: {:?}", data);
        assert_approx_eq!(
            buckets.supported_kl_div(&buckets_posterior).unwrap(),
            buckets.posterior_kl_div(&data).unwrap()
        );
        assert_approx_eq!(buckets.supported_kl_div(&buckets_posterior).unwrap(), 0.0);
    }

    #[test]
    fn dirichlet_kl_sanity_test() {
        let bucket1 = Dirichlet::from(&[(0, 6.0), (1, 6.0)][..]);
        println!("{:?}", bucket1);

        let bucket2 = Dirichlet::from(&[(0, 3.0), (1, 9.0)][..]);
        println!("{:?}", bucket2);

        let bucket3 = Dirichlet::from(&[(0, 5.5), (1, 6.5)][..]);
        println!("{:?}", bucket3);
        println!(
            "{:?}, {}",
            bucket1.supported_kl_div(&bucket2).unwrap(),
            bucket1.supported_kl_div(&bucket3).unwrap()
        );
        assert!(
            bucket1.supported_kl_div(&bucket2).unwrap()
                > bucket1.supported_kl_div(&bucket3).unwrap()
        );
    }

    #[test]
    fn dirichlet_aic_sanity_test() {
        let data1 = DiscreteData::from(&[(0, 4.0), (1, 0.0)][..]);
        println!("{:?}", data1);

        let data2 = DiscreteData::from(&[(0, 2.0), (1, 2.0)][..]);
        println!("{:?}", data2);

        let mut posterior = Dirichlet::new();
        posterior.add_observations(&data2);
        println!("{:?}", posterior);

        println!(
            "{:?}, {}",
            posterior.marginal_aic(&data1).unwrap(),
            posterior.marginal_aic(&data2).unwrap()
        );
        assert!(posterior.marginal_aic(&data1).unwrap() > posterior.marginal_aic(&data2).unwrap());
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
