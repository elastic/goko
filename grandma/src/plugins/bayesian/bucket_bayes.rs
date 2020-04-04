//! # Bucket probability
//!
//! A class for handling the finite probablity distribution of the children
use crate::node::CoverNode;
use crate::plugins::*;
use crate::tree::CoverTreeReader;

use super::*;
use std::fmt;

use rand::{thread_rng, Rng};
use statrs::function::gamma::{digamma, ln_gamma};
use std::collections::{HashMap, VecDeque};

/// Simple probability density function for where things go by count
///
#[derive(Debug, Clone)]
pub struct Dirichlet {
    child_counts: Vec<(NodeAddress, f64)>,
    singleton_count: f64,
}

impl Dirichlet {
    pub fn new() -> Dirichlet {
        Dirichlet {
            child_counts: Vec::new(),
            singleton_count: 0.0,
        }
    }

    pub fn weight(mut self, weight: f64) -> Dirichlet {
        self.child_counts.iter_mut().for_each(|(_, p)| *p *= weight);
        self.singleton_count *= weight;
        self
    }
    pub fn total(&self) -> f64 {
        self.singleton_count
            + self
                .child_counts
                .iter()
                .map(|(_, c)| c)
                .fold(0.0, |x, a| x + a)
    }
    pub fn add_child_pop(&mut self, loc: Option<NodeAddress>, count: f64) {
        match loc {
            Some(ca) => match self.child_counts.binary_search_by_key(&ca, |&(a, _)| a) {
                Ok(index) => self.child_counts[index].1 += count,
                Err(index) => self.child_counts.insert(index, (ca, count)),
            },
            None => self.singleton_count += count,
        }
    }

    pub fn remove_child_pop(&mut self, loc: Option<NodeAddress>, count: f64) {
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
impl BayesianDistribution for Dirichlet {
    fn add_observation(&mut self, loc: Option<NodeAddress>) {
        self.add_child_pop(loc, 1.0);
    }

    fn ln_prob(&self, loc: Option<NodeAddress>) -> f64 {
        let ax = match loc {
            Some(ca) => self
                .child_counts
                .binary_search_by_key(&ca, |&(a, _)| a)
                .map(|i| self.child_counts[i].1)
                .unwrap_or(0.0),
            None => self.singleton_count,
        };
        ax.ln() - self.total().ln()
    }
    /// from http://bariskurt.com/kullback-leibler-divergence-between-two-dirichlet-and-beta-distributions/
    /// We assume that the Dirichlet distribution passed into this one is conditioned on this one! It assumes they have the same keys!
    fn kl_divergence(&self, other: &Dirichlet) -> f64 {
        let mut my_total = self.total();
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
            0.0
        } else {
            kld
        }
    }
}

impl<M: Metric> NodePlugin<M> for Dirichlet {
    fn update(&mut self, _my_node: &CoverNode<M>, _my_tree: &CoverTreeReader<M>) {}
}

/// Zero sized type that can be passed around. Equivilant to `()`
#[derive(Debug, Clone)]
pub struct DirichletTree {}

impl<M: Metric> TreePlugin<M> for DirichletTree {
    fn update(&mut self, _my_tree: &CoverTreeReader<M>) {}
}

/// Zero sized type that can be passed around. Equivilant to `()`
#[derive(Debug, Clone)]
pub struct GrandmaDirichlet {}

/// Parent trait that make this all work. Ideally this should be included in the `TreePlugin` but rust doesn't like it.
impl<M: Metric> GrandmaPlugin<M> for GrandmaDirichlet {
    type NodeComponent = Dirichlet;
    type TreeComponent = DirichletTree;
    fn node_component(
        _parameters: &Self::TreeComponent,
        my_node: &CoverNode<M>,
        my_tree: &CoverTreeReader<M>,
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
            bucket.add_child_pop(None, my_node.singleton_len() as f64);
        } else {
            bucket.add_child_pop(None, (my_node.singleton_len() + 1) as f64);
        }
        bucket
    }
}

/// Computes a frequentist KL divergence calculation on each node the sequence touches.
pub struct BayesCategoricalTracker<M: Metric> {
    running_pdfs: HashMap<NodeAddress, Dirichlet>,
    sequence: VecDeque<Vec<NodeAddress>>,
    length: usize,
    prior_weight: f64,
    observation_weight: f64,
    reader: CoverTreeReader<M>,
}

impl<M: Metric> fmt::Debug for BayesCategoricalTracker<M> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "PointCloud {{ sequence: {:?}, length: {} prior_weight: {}, observation_weight: {}, running_pdfs: {:#?}}}",
            self.sequence, self.length, self.prior_weight, self.observation_weight, self.running_pdfs,
        )
    }
}

impl<M: Metric> BayesCategoricalTracker<M> {
    /// Creates a new blank thing with capacity `size`, input 0 for unlimited.
    pub fn new(
        prior_weight: f64,
        observation_weight: f64,
        size: usize,
        reader: CoverTreeReader<M>,
    ) -> BayesCategoricalTracker<M> {
        BayesCategoricalTracker {
            running_pdfs: HashMap::new(),
            sequence: VecDeque::new(),
            length: size,
            prior_weight,
            observation_weight,
            reader,
        }
    }
    fn add_trace_to_pdfs(&mut self, trace: &[NodeAddress]) {
        let parent_address_iter = trace.iter();
        let mut child_address_iter = trace.iter();
        child_address_iter.next();
        for (parent, child) in parent_address_iter.zip(child_address_iter) {
            self.running_pdfs
                .entry(*parent)
                .or_insert(
                    self.reader
                        .get_node_plugin_and::<Dirichlet, _, _>(*parent, |p| p.clone())
                        .unwrap()
                        .weight(self.prior_weight),
                )
                .add_child_pop(Some(*child), self.observation_weight);
        }
        let last = trace.last().unwrap();
        self.running_pdfs
            .entry(*last)
            .or_insert(
                self.reader
                    .get_node_plugin_and::<Dirichlet, _, _>(*last, |p| p.clone())
                    .unwrap()
                    .weight(self.prior_weight),
            )
            .add_child_pop(None, self.observation_weight);
    }

    fn remove_trace_from_pdfs(&mut self, trace: &[NodeAddress]) {
        let parent_address_iter = trace.iter();
        let mut child_address_iter = trace.iter();
        child_address_iter.next();
        for (parent, child) in parent_address_iter.zip(child_address_iter) {
            self.running_pdfs
                .entry(*parent)
                .or_insert(
                    self.reader
                        .get_node_plugin_and::<Dirichlet, _, _>(*parent, |p| p.clone())
                        .unwrap()
                        .weight(self.prior_weight),
                )
                .remove_child_pop(Some(*child), self.observation_weight);
        }
        let last = trace.last().unwrap();
        self.running_pdfs
            .entry(*last)
            .or_insert(
                self.reader
                    .get_node_plugin_and::<Dirichlet, _, _>(*last, |p| p.clone())
                    .unwrap()
                    .weight(self.prior_weight),
            )
            .remove_child_pop(None, self.observation_weight);
    }
}
impl<M: Metric> InsertDistributionTracker<M> for BayesCategoricalTracker<M> {
    type Distribution = Dirichlet;
    /// Adds an element to the trace
    fn add_trace(&mut self, trace: Vec<NodeAddress>) {
        self.add_trace_to_pdfs(&trace);
        self.sequence.push_back(trace);
        if self.sequence.len() > self.length && self.length != 0 {
            let oldest = self.sequence.pop_front().unwrap();
            self.remove_trace_from_pdfs(&oldest);
        }
    }
    fn running_pdfs(&self) -> &HashMap<NodeAddress, Dirichlet> {
        &self.running_pdfs
    }
    fn tree_reader(&self) -> &CoverTreeReader<M> {
        &self.reader
    }
    fn sequence_len(&self) -> usize {
        self.sequence.len()
    }
}

pub struct DirichletBaseline<M: Metric> {
    sequence_len: usize,
    sequence_count: usize,
    sequence_cap: usize,
    prior_weight: f64,
    observation_weight: f64,
    reader: CoverTreeReader<M>,
}

impl<M: Metric> DirichletBaseline<M> {
    pub fn new(reader: CoverTreeReader<M>) -> DirichletBaseline<M> {
        DirichletBaseline {
            sequence_len: 200,
            sequence_count: 100,
            sequence_cap: 100,
            prior_weight: 1.0,
            observation_weight: 1.0,
            reader,
        }
    }

    pub fn set_sequence_len(&mut self, sequence_len: usize) {
        self.sequence_len = sequence_len;
    }
    pub fn set_sequence_count(&mut self, sequence_count: usize) {
        self.sequence_count = sequence_count;
    }
    pub fn set_sequence_cap(&mut self, sequence_cap: usize) {
        self.sequence_cap = sequence_cap;
    }
    pub fn set_prior_weight(&mut self, prior_weight: f64) {
        self.prior_weight = prior_weight;
    }
    pub fn set_observation_weight(&mut self, observation_weight: f64) {
        self.observation_weight = observation_weight;
    }

    pub fn train(&self) -> GrandmaResult<Vec<Vec<KLDivergenceStats>>> {
        /*
        let chunk_size = 10;
        let (results_sender, results_receiver): (
            Sender<KLDivergenceStats>,
            Receiver<KLDivergenceStats>,
        ) = unbounded();
        */
        let mut results: Vec<Vec<KLDivergenceStats>> = (0..self.sequence_count)
            .map(|_| Vec::with_capacity(self.sequence_len))
            .collect();
        let point_cloud = self.reader.point_cloud();
        for i in 0..self.sequence_count {
            let mut tracker = BayesCategoricalTracker::new(
                self.prior_weight,
                self.observation_weight,
                self.sequence_cap,
                self.reader.clone(),
            );
            for _ in 0..self.sequence_len {
                let mut rng = thread_rng();
                let query_point =
                    point_cloud.get_point(rng.gen_range(0, point_cloud.len()) as u64)?;
                tracker.add_trace(
                    self.reader
                        .dry_insert(query_point)?
                        .iter()
                        .map(|(_, a)| *a)
                        .collect(),
                );
                results[i].push(tracker.current_stats());
            }
        }

        Ok(results)
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
        assert_approx_eq!(buckets.ln_prob(None), 0.0);
        assert_approx_eq!(buckets.kl_divergence(&buckets), 0.0);
    }

    #[test]
    fn dirichlet_mixed_sanity_test() {
        let mut buckets = Dirichlet::new();
        buckets.add_child_pop(None, 5.0);
        buckets.add_child_pop(Some((0, 0)), 5.0);
        println!("{:?}", buckets);
        assert_approx_eq!(buckets.ln_prob(None), 0.5f64.ln());
        assert_approx_eq!(buckets.ln_prob(Some((0, 0))), 0.5f64.ln());
        assert_approx_eq!(buckets.kl_divergence(&buckets), 0.0);
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
            bucket1.kl_divergence(&bucket2),
            bucket1.kl_divergence(&bucket3)
        );
        assert!(bucket1.kl_divergence(&bucket2) > bucket1.kl_divergence(&bucket3));
    }


}
