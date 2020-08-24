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

use super::*;
use std::fmt;

use rand::prelude::*;
use rayon::iter::repeatn;
use statrs::function::gamma::{digamma, ln_gamma};
use std::collections::{HashMap, VecDeque};

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
    pub fn prob_vector(&self) -> Option<(Vec<(NodeAddress,f64)>,f64)> {

        let total = self.total();
        if total > 0.0 {
            let v: Vec<(NodeAddress,f64)> = self.child_counts.iter().map(|(na,f)| (*na,f/total)).collect();
            Some((v,self.singleton_count/total))
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
}
impl DiscreteBayesianDistribution for Dirichlet {
    type Evidence = Categorical;

    fn add_observation(&mut self, loc: Option<NodeAddress>) {
        self.add_child_pop(loc, 1.0);
    }

    fn add_evidence(&mut self, other: &Categorical) {
        for (na, c) in &other.child_counts {
            self.add_child_pop(Some(*na), *c);
        }
        self.add_child_pop(None, other.singleton_count);
    }

    fn posterior_kl_divergence(&self, other: &Categorical) -> Option<f64> {
        let my_total = self.total();
        let other_total = other.total() + my_total;
        let mut my_total_lng = 0.0;
        let mut other_total_lng = 0.0;
        let mut digamma_portion = 0.0;
        if self.singleton_count > 0.0 {
            other_total_lng += ln_gamma(other.singleton_count + self.singleton_count);
            my_total_lng += ln_gamma(self.singleton_count);
            digamma_portion -= other.singleton_count
                * (digamma(self.singleton_count) - digamma(my_total));
        }
        for (other_ca, other_ca_count) in other.child_counts.iter()
        {
            let ca_count = match self.child_counts.binary_search_by_key(other_ca, |&(a, _)| a) {
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
}
impl DiscreteDistribution for Dirichlet {
    fn ln_prob(&self, loc: Option<&NodeAddress>) -> Option<f64> {
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
    /// from http://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/
    /// We assume that the Dirichlet distribution passed into this one is conditioned on this one! It assumes they have the same keys!
    fn kl_divergence(&self, other: &Dirichlet) -> Option<f64> {
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

impl<D: PointCloud> NodePlugin<D> for Dirichlet {
    fn update(&mut self, _my_node: &CoverNode<D>, _my_tree: &CoverTreeReader<D>) {}
}

/// Zero sized type that can be passed around. Equivilant to `()`
#[derive(Debug, Clone)]
pub struct DirichletTree {}

impl<D: PointCloud> TreePlugin<D> for DirichletTree {
    fn update(&mut self, _my_tree: &CoverTreeReader<D>) {}
}

/// Zero sized type that can be passed around. Equivilant to `()`
#[derive(Debug, Clone)]
pub struct GokoDirichlet {}

/// Parent trait that make this all work. Ideally this should be included in the `TreePlugin` but rust doesn't like it.
impl<D: PointCloud> GokoPlugin<D> for GokoDirichlet {
    type NodeComponent = Dirichlet;
    type TreeComponent = DirichletTree;
    fn node_component(
        _parameters: &Self::TreeComponent,
        my_node: &CoverNode<D>,
        my_tree: &CoverTreeReader<D>,
    ) -> Self::NodeComponent {
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
        bucket
    }
}

/// Computes a frequentist KL divergence calculation on each node the sequence touches.
pub struct BayesCategoricalTracker<D: PointCloud> {
    running_evidence: HashMap<NodeAddress, Categorical>,
    sequence_queue: VecDeque<Vec<(f32, NodeAddress)>>,
    sequence_count: usize,
    window_size: usize,
    prior_weight: f64,
    observation_weight: f64,
    reader: CoverTreeReader<D>,
}

impl<D: PointCloud> fmt::Debug for BayesCategoricalTracker<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "PointCloud {{ sequence_queue: {:?}, window_size: {} prior_weight: {}, observation_weight: {}, running_evidence: {:?}}}",
            self.sequence_queue, self.window_size, self.prior_weight, self.observation_weight, self.running_evidence,
        )
    }
}

impl<D: PointCloud> BayesCategoricalTracker<D> {
    /// Creates a new blank thing with capacity `size`, input 0 for unlimited.
    pub fn new(
        prior_weight: f64,
        observation_weight: f64,
        window_size: usize,
        reader: CoverTreeReader<D>,
    ) -> BayesCategoricalTracker<D> {
        BayesCategoricalTracker {
            running_evidence: HashMap::new(),
            sequence_queue: VecDeque::new(),
            sequence_count: 0,
            window_size,
            prior_weight,
            observation_weight,
            reader,
        }
    }

    /// Appends a tracker to this one,
    pub fn append(mut self, other: &Self) -> Self {
        for (k, v) in other.running_evidence.iter() {
            self.running_evidence
                .entry(*k)
                .and_modify(|e| e.merge(v))
                .or_insert(v.clone());
        }
        self.sequence_queue.extend(other.sequence_queue.iter().cloned());
        self
    }

    fn get_distro(&self, address: NodeAddress) -> Dirichlet {
        let mut prob = self
            .reader
            .get_node_plugin_and::<Dirichlet, _, _>(address, |p| p.clone())
            .unwrap();
        let total = prob.total();
        if total > self.window_size as f64 {
            prob.weight((total.ln() * self.window_size as f64) / total)
        }
        prob.weight(self.prior_weight);
        prob
    }

    fn add_trace_to_pdfs(&mut self, trace: &[(f32, NodeAddress)]) {
        let parent_address_iter = trace.iter().map(|(_, ca)| ca);
        let mut child_address_iter = trace.iter().map(|(_, ca)| ca);
        child_address_iter.next();
        for (parent, child) in parent_address_iter.zip(child_address_iter) {
            self.running_evidence
                .entry(*parent)
                .or_default()
                .add_child_pop(Some(*child), self.observation_weight);
        }
        let last = trace.last().unwrap().1;
        self.running_evidence
            .entry(last)
            .or_default()
            .add_child_pop(None, self.observation_weight);
    }

    fn remove_trace_from_pdfs(&mut self, trace: &[(f32, NodeAddress)]) {
        let parent_address_iter = trace.iter().map(|(_, ca)| ca);
        let mut child_address_iter = trace.iter().map(|(_, ca)| ca);
        child_address_iter.next();
        for (parent, child) in parent_address_iter.zip(child_address_iter) {
            let parent_evidence = self.running_evidence
                .get_mut(parent)
                .unwrap();
            parent_evidence.remove_child_pop(Some(*child), self.observation_weight);
        }
        let last = trace.last().unwrap().1;
        self.running_evidence
            .get_mut(&last)
            .unwrap()
            .remove_child_pop(None, self.observation_weight);
    }

    /// Gives the probability vector for this
    pub fn prob_vector(&self,na:NodeAddress) -> Option<(Vec<(NodeAddress,f64)>,f64)> {
        self.reader
            .get_node_plugin_and::<Dirichlet, _, _>(na, |p| {
                let mut dir = p.clone();
                if let Some(e) = self.running_evidence.get(&na) {
                    dir.add_evidence(e)
                }
                dir.prob_vector()
            }).flatten()
    }

    /// Gives the probability vector for this
    pub fn evidence_prob_vector(&self,na:NodeAddress) -> Option<(Vec<(NodeAddress,f64)>,f64)> {
        self.running_evidence.get(&na).map(|e| e.prob_vector()).flatten()
    }
}
impl<D: PointCloud> DiscreteBayesianSequenceTracker<D> for BayesCategoricalTracker<D> {
    type Distribution = Dirichlet;
    /// Adds an element to the trace
    fn add_path(&mut self, trace: Vec<(f32, NodeAddress)>) {
        self.add_trace_to_pdfs(&trace);
        self.sequence_count += 1;
        if self.window_size != 0 {
            self.sequence_queue.push_back(trace);

            if self.sequence_queue.len() > self.window_size {
                let oldest = self.sequence_queue.pop_front().unwrap();
                self.remove_trace_from_pdfs(&oldest);
            }
        }
    }
    fn running_evidence(&self) -> &HashMap<NodeAddress, Categorical> {
        &self.running_evidence
    }
    fn tree_reader(&self) -> &CoverTreeReader<D> {
        &self.reader
    }
    fn sequence_len(&self) -> usize {
        if self.sequence_queue.len() > 0 {
            self.sequence_queue.len()
        } else {
            self.sequence_count
        }
    }
}

/// Trains a baseline by sampling randomly from the training set (used to create the tree)
/// This baseline is _not_ realistic.
pub struct DirichletBaseline {
    sample_rate: usize,
    sequence_len: usize,
    num_sequences: usize,
    prior_weight: f64,
    observation_weight: f64,
}

impl DirichletBaseline {
    /// New with sensible defaults
    pub fn new() -> DirichletBaseline {
        DirichletBaseline {
            sample_rate: 100,
            sequence_len: 0,
            num_sequences: 8,
            prior_weight: 1.0,
            observation_weight: 1.0,
        }
    }
    /// Sets a new maxium sequence length. Set this to be the window size if you're using windows, the lenght of the test set you've got,
    /// or leave it alone as the default limit is the number of points in the training set. 
    ///
    /// We sample up to this cap, linearly interpolating above this. So, the baseline produced is fairly accurate for indexes below this 
    /// and unreliable above this.
    pub fn set_sequence_len(&mut self, sequence_len: usize) {
        self.sequence_len = sequence_len;
    }
    /// Sets a new count of sequences to train over, default 100. Stats for each sequence are returned.
    pub fn set_num_sequences(&mut self, num_sequences: usize) {
        self.num_sequences = num_sequences;
    }
    /// Sets a new prior weight, default 1.0. The prior is multiplied by this to increase or decrease it's importance
    pub fn set_prior_weight(&mut self, prior_weight: f64) {
        self.prior_weight = prior_weight;
    }
    /// Sets a new observation weight, default 1.0. Each discrete observation is treated as having this value.
    pub fn set_observation_weight(&mut self, observation_weight: f64) {
        self.observation_weight = observation_weight;
    }
    /// Samples at the following rate, then interpolates for sequence lengths between the following. 
    pub fn set_sample_rate(&mut self, sample_rate: usize) {
        self.sample_rate = sample_rate;
    }

    /// Trains the sequences up.
    pub fn train<D: PointCloud>(&self,reader: CoverTreeReader<D>) -> GokoResult<KLDivergenceBaseline> {
        let point_indexes = reader.point_cloud().reference_indexes();
        let sequence_len = if self.sequence_len == 0 {
            point_indexes.len()
        } else {
            self.sequence_len
        };

        let results: Vec<Vec<KLDivergenceStats>> = repeatn(reader.clone(), self.num_sequences).map(|reader| {
            let mut tracker = BayesCategoricalTracker::new(
                self.prior_weight,
                self.observation_weight,
                0,
                reader,
            );
            (&point_indexes[..]).choose_multiple(&mut thread_rng(),sequence_len).enumerate().filter_map(|(i,pi)| {
                tracker.add_path(tracker.reader.known_path(*pi).unwrap());
                if i % self.sample_rate == 0 {
                    Some(tracker.kl_div_stats())
                } else {
                    None
                }
            }).collect()
        }).collect();
        let len = results[0].len();
        let mut sequence_len = Vec::with_capacity(len);
        let mut stats: Vec<KLDivergenceBaselineStats> = std::iter::repeat_with(|| KLDivergenceBaselineStats::default()).take(len).collect();

        for i in 0..len {
            for result_vec in &results {
                stats[i].add(&result_vec[i]);
            }
            sequence_len.push(results[0][i].sequence_len);
        }
        Ok(KLDivergenceBaseline {
            num_sequences: results.len(),
            stats,
            sequence_len,
        })
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    //use crate::tree::tests::build_basic_tree;

    #[test]
    fn dirichlet_sanity_test() {
        let mut buckets = Dirichlet::new();
        buckets.add_child_pop(None, 5.0);
        println!("{:?}", buckets);
        assert_approx_eq!(buckets.ln_prob(None).unwrap(), 0.0);
        assert_approx_eq!(buckets.kl_divergence(&buckets).unwrap(), 0.0);
    }

    #[test]
    fn dirichlet_mixed_sanity_test() {
        let mut buckets = Dirichlet::new();
        buckets.add_child_pop(None, 5.0);
        buckets.add_child_pop(Some((0, 0)), 5.0);
        println!("{:?}", buckets);
        assert_approx_eq!(buckets.ln_prob(None).unwrap(), 0.5f64.ln());
        assert_approx_eq!(buckets.ln_prob(Some(&(0, 0))).unwrap(), 0.5f64.ln());
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
        assert_approx_eq!(buckets_posterior.ln_prob(None).unwrap(), 0.5f64.ln());
        assert_approx_eq!(buckets_posterior.ln_prob(Some(&(0, 0))).unwrap(), 0.5f64.ln());
        assert_approx_eq!(buckets.kl_divergence(&buckets_posterior).unwrap(), buckets.posterior_kl_divergence(&categorical).unwrap());
    }

    #[test]
    fn dirichlet_posterior_sanity_test_2() {
        let mut buckets = Dirichlet::new();
        buckets.add_child_pop(None, 3.0);
        buckets.add_child_pop(Some((0, 0)), 2.0);

        let mut categorical = Categorical::new();
        categorical.add_child_pop(None,3.0);
        categorical.add_child_pop(Some((0, 0)), 2.0);

        let mut buckets_posterior = buckets.clone();
        buckets_posterior.add_evidence(&categorical);

        println!("Buckets: {:?}", buckets);
        println!("Buckets Posterior: {:?}", buckets_posterior);
        println!("Evidence: {:?}", categorical);
        assert_approx_eq!(buckets.kl_divergence(&buckets_posterior).unwrap(), buckets.posterior_kl_divergence(&categorical).unwrap());
        assert_approx_eq!(buckets.kl_divergence(&buckets_posterior).unwrap(), 0.1789970483832892);
    }

    #[test]
    fn dirichlet_posterior_sanity_test_3() {
        let mut buckets = Dirichlet::new();
        buckets.add_child_pop(None, 3.0);

        let mut categorical = Categorical::new();
        categorical.add_child_pop(None,3.0);

        let mut buckets_posterior = buckets.clone();
        buckets_posterior.add_evidence(&categorical);

        println!("Buckets: {:?}", buckets);
        println!("Buckets Posterior: {:?}", buckets_posterior);
        println!("Evidence: {:?}", categorical);
        assert_approx_eq!(buckets.kl_divergence(&buckets_posterior).unwrap(), buckets.posterior_kl_divergence(&categorical).unwrap());
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
}
