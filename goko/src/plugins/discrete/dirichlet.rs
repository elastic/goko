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

use crate::covertree::node::CoverNode;
use crate::covertree::CoverTreeReader;
use crate::plugins::*;

use rand::prelude::*;
use statrs::function::gamma::{digamma, ln_gamma};

use rand::distributions::{Distribution, Uniform};

use super::categorical::*;

/// Simple probability density function for where things go by count
///
#[derive(Debug, Clone, Default)]
pub struct Dirichlet {
    child_counts: Vec<(NodeAddress, f64)>,
    singleton_count: f64,
}

impl Dirichlet {
    /// New all 0 Dirichlet distribution. The child counts are uninitialized
    pub fn new() -> Dirichlet {
        Dirichlet {
            child_counts: Vec::new(),
            singleton_count: 0.0,
        }
    }
    /// Multiplies all parameters by this weight
    pub fn weight(&mut self, weight: f64) {
        self.child_counts.iter_mut().for_each(|(_, p)| *p *= weight);
        self.singleton_count *= weight;
    }
    /// The total of the parameters. This is a proxy for the total count, and the "concentration" of the distribution
    pub fn total(&self) -> f64 {
        self.singleton_count
            + self
                .child_counts
                .iter()
                .map(|(_, c)| c)
                .fold(0.0, |x, a| x + a)
    }

    /// Gives the probability vector for this
    pub fn prob_vector(&self) -> Option<(Vec<(NodeAddress, f64)>, f64)> {
        let total = self.total();
        if total > 0.0 {
            let v: Vec<(NodeAddress, f64)> = self
                .child_counts
                .iter()
                .map(|(na, f)| (*na, f / total))
                .collect();
            Some((v, self.singleton_count / total))
        } else {
            None
        }
    }

    /// Gives the probability vector for this
    pub fn ln_prob_vector(&self) -> Option<(Vec<(NodeAddress, f64)>, f64)> {
        let total_ln = self.total().ln();
        if total_ln > 1.0 {
            let v: Vec<(NodeAddress, f64)> = self
                .child_counts
                .iter()
                .map(|(na, f)| (*na, f.ln() - total_ln))
                .collect();
            Some((v, self.singleton_count.ln() - total_ln))
        } else {
            None
        }
    }

    fn add_child_pop(&mut self, loc: Option<NodeAddress>, count: f64) {
        match loc {
            Some(ca) => match self.child_counts.binary_search_by_key(&ca, |&(a, _)| a) {
                Ok(index) => self.child_counts[index].1 += count,
                Err(index) => self.child_counts.insert(index, (ca, count)),
            },
            None => self.singleton_count += count,
        }
    }

    fn remove_child_pop(&mut self, loc: Option<NodeAddress>, count: f64) {
        match loc {
            Some(ca) => {
                if let Ok(index) = self.child_counts.binary_search_by_key(&ca, |&(a, _)| a) {
                    if self.child_counts[index].1 < count {
                        self.child_counts[index].1 = 0.0;
                    } else {
                        self.child_counts[index].1 -= count;
                    }
                }
            }
            None => {
                if self.singleton_count < count as f64 {
                    self.singleton_count = 0.0;
                } else {
                    self.singleton_count -= count as f64;
                }
            }
        }
    }

    /// Adds a single observation to the Dirichlet distribution.
    /// Mutates the distribution in place to the posterior given the new evidence.
    pub fn add_observation(&mut self, loc: Option<NodeAddress>) {
        self.add_child_pop(loc, 1.0);
    }

    /// Adds a a group of observations to the Dirichlet distribution.
    /// Mutates the distribution in place to the posterior given the new evidence.
    pub fn add_evidence(&mut self, other: &Categorical) {
        for (na, c) in &other.child_counts {
            self.add_child_pop(Some(*na), *c);
        }
        self.add_child_pop(None, other.singleton_count);
    }

    /// Computes KL(prior || posterior), where the prior is the distribution
    /// and the posterior is based on the evidence provided.
    pub fn posterior_kl_divergence(&self, other: &Categorical) -> Option<f64> {
        let my_total = self.total();
        let other_total = other.total() + my_total;
        let mut my_total_lng = 0.0;
        let mut other_total_lng = 0.0;
        let mut digamma_portion = 0.0;
        if self.singleton_count > 0.0 {
            other_total_lng += ln_gamma(other.singleton_count + self.singleton_count);
            my_total_lng += ln_gamma(self.singleton_count);
            digamma_portion -=
                other.singleton_count * (digamma(self.singleton_count) - digamma(my_total));
        }
        for (other_ca, other_ca_count) in other.child_counts.iter() {
            let ca_count = match self
                .child_counts
                .binary_search_by_key(other_ca, |&(a, _)| a)
            {
                Ok(ind) => self.child_counts[ind].1,
                Err(_) => return None,
            };

            my_total_lng += ln_gamma(ca_count);
            other_total_lng += ln_gamma(*other_ca_count + ca_count);
            digamma_portion -= *other_ca_count * (digamma(ca_count) - digamma(my_total));
        }

        let kld = ln_gamma(my_total) - my_total_lng - ln_gamma(other_total)
            + other_total_lng
            + digamma_portion;
        // for floating point errors, sometimes this is -0.000000001
        if kld < 0.0 {
            Some(0.0)
        } else {
            Some(kld)
        }
    }

    /// Computes the log of the expected PDF of the Dirichlet distribution
    pub fn ln_pdf(&self, loc: Option<&NodeAddress>) -> Option<f64> {
        let ax = match loc {
            Some(ca) => self
                .child_counts
                .binary_search_by_key(&ca, |(a, _)| a)
                .map(|i| self.child_counts[i].1)
                .unwrap_or(0.0),
            None => self.singleton_count,
        };
        Some(ax.ln() - self.total().ln())
    }

    /// Samples from the expected PDF of the Dirichlet distribution
    pub fn sample<R: Rng>(&self, rng: &mut R) -> Option<NodeAddress> {
        let sum = self.total() as usize;
        let uniform = Uniform::from(0..sum);
        let sample = uniform.sample(rng) as f64;

        let mut count = 0.0;
        for (a, c) in &self.child_counts {
            count += c;
            if sample < count {
                return Some(*a);
            }
        }
        None
    }

    /// from http://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/
    /// We assume that the Dirichlet distribution passed into this one is conditioned on this one! It assumes they have the same keys!
    pub fn kl_divergence(&self, other: &Dirichlet) -> Option<f64> {
        let my_total = self.total();
        let mut other_total = other.singleton_count;
        let mut my_total_lng = 0.0;
        let mut other_total_lng = 0.0;
        let mut digamma_portion = 0.0;
        if self.singleton_count > 0.0 {
            other_total_lng += ln_gamma(other.singleton_count);
            my_total_lng += ln_gamma(self.singleton_count);
            digamma_portion += (self.singleton_count - other.singleton_count)
                * (digamma(self.singleton_count) - digamma(my_total));
        }
        for ((ca, ca_count), (other_ca, other_ca_count)) in
            self.child_counts.iter().zip(other.child_counts.iter())
        {
            assert_eq!(ca, other_ca);
            other_total += *other_ca_count;
            my_total_lng += ln_gamma(*ca_count);
            other_total_lng += ln_gamma(*other_ca_count);
            digamma_portion +=
                (*ca_count - *other_ca_count) * (digamma(*ca_count) - digamma(my_total));
        }

        let kld = ln_gamma(my_total) - my_total_lng - ln_gamma(other_total)
            + other_total_lng
            + digamma_portion;
        // for floating point errors, sometimes this is -0.000000001
        if kld < 0.0 {
            Some(0.0)
        } else {
            Some(kld)
        }
    }
}

impl<D: PointCloud> NodePlugin<D> for Dirichlet {}

/// Stores the log probabilities for each node in the tree. 
/// 
/// This is the probability that when you sample from the tree you end up at a particular node.
#[derive(Debug, Clone, Default)]
pub struct GokoDirichlet {
    // probability that you'd pass thru this node.
    //pub cond_ln_probs: HashMap<NodeAddress,f64>,
}

/// Parent trait that make this all work. Ideally this should be included in the `TreePlugin` but rust doesn't like it.
impl<D: PointCloud> GokoPlugin<D> for GokoDirichlet {
    type NodeComponent = Dirichlet;
    fn node_component(
        _parameters: &Self,
        my_node: &CoverNode<D>,
        my_tree: &CoverTreeReader<D>,
    ) -> Option<Self::NodeComponent> {
        let mut bucket = Dirichlet::new();

        // If we're a routing node then grab the childen's values
        if let Some((nested_scale, child_addresses)) = my_node.children() {
            my_tree.get_node_plugin_and::<Self::NodeComponent, _, _>(
                (nested_scale, *my_node.center_index()),
                |p| {
                    bucket.add_child_pop(Some((nested_scale, *my_node.center_index())), p.total());
                },
            );
            for ca in child_addresses {
                my_tree.get_node_plugin_and::<Self::NodeComponent, _, _>(*ca, |p| {
                    bucket.add_child_pop(Some(*ca), p.total());
                });
            }
            bucket.add_child_pop(None, my_node.singletons_len() as f64);
        } else {
            bucket.add_child_pop(None, (my_node.singletons_len() + 1) as f64);
        }
        Some(bucket)
    }

    /*
    fn tree_component(parameters: &mut Self, my_tree: &mut CoverTreeWriter<D>) {
        let mut unvisited_nodes = vec![my_tree.root_address];
        let reader = my_tree.reader();
        parameters.cond_ln_probs.insert(my_tree.root_address, 0.0);
        while let Some(addr) = unvisited_nodes.pop() {
            let pass_thru_prob = parameters.cond_ln_probs.get(&addr).unwrap().clone();
            let ln_probs = reader.get_node_plugin_and::<Self::NodeComponent,_,_>(addr, |p| p.ln_prob_vector()).unwrap();
            if let Some((child_probs,_singleton_prob)) = ln_probs {
                for (child_addr,child_prob) in child_probs {
                    parameters.cond_ln_probs.insert(child_addr, pass_thru_prob + child_prob);
                    unvisited_nodes.push(child_addr);
                }
            }
        }
    }
    */
}



#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::covertree::tests::build_basic_tree;

    #[test]
    fn dirichlet_sanity_test() {
        let mut buckets = Dirichlet::new();
        buckets.add_child_pop(None, 5.0);
        println!("{:?}", buckets);
        assert_approx_eq!(buckets.ln_pdf(None).unwrap(), 0.0);
        assert_approx_eq!(buckets.kl_divergence(&buckets).unwrap(), 0.0);
    }

    #[test]
    fn dirichlet_mixed_sanity_test() {
        let mut buckets = Dirichlet::new();
        buckets.add_child_pop(None, 5.0);
        buckets.add_child_pop(Some((0, 0)), 5.0);
        println!("{:?}", buckets);
        assert_approx_eq!(buckets.ln_pdf(None).unwrap(), 0.5f64.ln());
        assert_approx_eq!(buckets.ln_pdf(Some(&(0, 0))).unwrap(), 0.5f64.ln());
        assert_approx_eq!(buckets.kl_divergence(&buckets).unwrap(), 0.0);
    }

    #[test]
    fn dirichlet_posterior_sanity_test() {
        let mut buckets = Dirichlet::new();
        buckets.add_child_pop(None, 3.0);
        buckets.add_child_pop(Some((0, 0)), 5.0);

        let mut categorical = Categorical::new();
        categorical.add_child_pop(None, 2.0);

        let mut buckets_posterior = buckets.clone();
        buckets_posterior.add_evidence(&categorical);

        println!("Buckets: {:?}", buckets);
        println!("Buckets Posterior: {:?}", buckets_posterior);
        println!("Evidence: {:?}", categorical);
        assert_approx_eq!(buckets_posterior.ln_pdf(None).unwrap(), 0.5f64.ln());
        assert_approx_eq!(
            buckets_posterior.ln_pdf(Some(&(0, 0))).unwrap(),
            0.5f64.ln()
        );
        assert_approx_eq!(
            buckets.kl_divergence(&buckets_posterior).unwrap(),
            buckets.posterior_kl_divergence(&categorical).unwrap()
        );
    }

    #[test]
    fn dirichlet_posterior_sanity_test_2() {
        let mut buckets = Dirichlet::new();
        buckets.add_child_pop(None, 3.0);
        buckets.add_child_pop(Some((0, 0)), 2.0);

        let mut categorical = Categorical::new();
        categorical.add_child_pop(None, 3.0);
        categorical.add_child_pop(Some((0, 0)), 2.0);

        let mut buckets_posterior = buckets.clone();
        buckets_posterior.add_evidence(&categorical);

        println!("Buckets: {:?}", buckets);
        println!("Buckets Posterior: {:?}", buckets_posterior);
        println!("Evidence: {:?}", categorical);
        assert_approx_eq!(
            buckets.kl_divergence(&buckets_posterior).unwrap(),
            buckets.posterior_kl_divergence(&categorical).unwrap()
        );
        assert_approx_eq!(
            buckets.kl_divergence(&buckets_posterior).unwrap(),
            0.1789970483832892
        );
    }

    #[test]
    fn dirichlet_posterior_sanity_test_3() {
        let mut buckets = Dirichlet::new();
        buckets.add_child_pop(None, 3.0);

        let mut categorical = Categorical::new();
        categorical.add_child_pop(None, 3.0);

        let mut buckets_posterior = buckets.clone();
        buckets_posterior.add_evidence(&categorical);

        println!("Buckets: {:?}", buckets);
        println!("Buckets Posterior: {:?}", buckets_posterior);
        println!("Evidence: {:?}", categorical);
        assert_approx_eq!(
            buckets.kl_divergence(&buckets_posterior).unwrap(),
            buckets.posterior_kl_divergence(&categorical).unwrap()
        );
        assert_approx_eq!(buckets.kl_divergence(&buckets_posterior).unwrap(), 0.0);
    }

    #[test]
    fn dirichlet_kl_sanity_test() {
        let mut bucket1 = Dirichlet::new();
        bucket1.add_child_pop(None, 6.0);
        bucket1.add_child_pop(Some((0, 0)), 6.0);
        println!("{:?}", bucket1);

        let mut bucket2 = Dirichlet::new();
        bucket2.add_child_pop(None, 3.0);
        bucket2.add_child_pop(Some((0, 0)), 9.0);
        println!("{:?}", bucket2);

        let mut bucket3 = Dirichlet::new();
        bucket3.add_child_pop(None, 5.5);
        bucket3.add_child_pop(Some((0, 0)), 6.5);
        println!("{:?}", bucket3);
        println!(
            "{:?}, {}",
            bucket1.kl_divergence(&bucket2).unwrap(),
            bucket1.kl_divergence(&bucket3).unwrap()
        );
        assert!(
            bucket1.kl_divergence(&bucket2).unwrap() > bucket1.kl_divergence(&bucket3).unwrap()
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
