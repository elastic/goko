//! See the paper for how this works

use crate::covertree::CoverTreeReader;
use crate::plugins::*;
use crate::NodeAddress;
use hashbrown::HashMap;

use serde::{Deserialize, Serialize};
pub use stats_goko::discrete::*;
pub use stats_goko::moments::Moments;
use std::fmt;

use std::collections::VecDeque;

use super::small::*;

/// Computes a frequentist KL divergence calculation on each node the sequence touches.
#[derive(Clone)]
pub struct BayesCovertree {
    overall_tracker: DirichletTracker,
    node_trackers: HashMap<NodeAddress, DirichletTracker>,
    sequence_queue: VecDeque<Vec<(NodeAddress, f32)>>,
    sequence_count: usize,
    window_size: usize,
}

impl fmt::Debug for BayesCovertree {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "PointCloud {{ sequence_queue: {:?}, window_size: {}, node_trackers: {:?}}}",
            self.sequence_queue, self.window_size, self.node_trackers,
        )
    }
}

impl BayesCovertree {
    /// Creates a new blank thing with capacity `size`, input 0 for unlimited.
    pub fn new<D: PointCloud>(window_size: usize, reader: &CoverTreeReader<D>) -> BayesCovertree {
        let total_alpha_overall =
            (reader.parameters().point_cloud.len() + reader.node_count()) as f64;
        let mut overall_tracker = DirichletTracker::sparse(total_alpha_overall, reader.node_count());
        let mut node_trackers = HashMap::with_capacity(reader.node_count());
        reader
            .layers()
            .for_each(|(_, l)| {
                l.for_each_node(|_, n| {
                    let a = n.address();
                    n.get_plugin_and::<Dirichlet, _, _>(|p| {
                        overall_tracker.set_alpha(
                            a,
                            p.get_alpha(NodeAddress::SINGLETON)
                                .expect("There should always be singletons."),
                        );
                        node_trackers.insert(a, p.tracker());
                    });
                })
            });
        BayesCovertree {
            overall_tracker,
            node_trackers,
            sequence_queue: VecDeque::with_capacity(window_size),
            sequence_count: 0,
            window_size,
        }
    }

    pub fn clone_new_window(&self, window_size: usize) -> Self {
        let mut clone = self.clone();
        clone.window_size = window_size;
        clone
    }

    pub fn node_count(&self) -> usize {
        self.node_trackers.len()
    }

    /// Create a smaller serializable version of this tracker.
    pub fn small(&self) -> BayesCovertreeSmall {
        let node_trackers = self
            .node_trackers
            .iter()
            .filter_map(|(i, v)| {
                let r = BayesNodeSmall {
                    kl_div: v.kl_div(),
                    mll: v.mll(),
                };
                if r.kl_div > 1e-7 || r.mll < -1e-7 {
                    Some((i.raw(), r))
                } else {
                    None
                }
            })
            .collect();
        let overall_tracker = BayesNodeSmall {
            kl_div: self.overall_tracker.kl_div(),
            mll: self.overall_tracker.mll(),
        };
        BayesCovertreeSmall {
            overall_tracker,
            node_trackers,
            sequence_count: self.sequence_count,
            window_size: self.window_size,
        }
    }

    pub fn marginal_posterior_probs(&self, na: NodeAddress) -> Option<Vec<(NodeAddress, f64)>> {
        self.node_trackers.get(&na).map(|t| t.marginal_posterior_probs())
    }

    fn add_trace(&mut self, trace: &[(NodeAddress, f32)]) {

        let parent_address_iter = trace.iter().map(|(ca, _)| ca);
        let mut child_address_iter = trace.iter().map(|(ca, _)| ca);
        child_address_iter.next();
 

        for (parent, child) in parent_address_iter.zip(child_address_iter) {
            let tracker = self.node_trackers.get_mut(parent).unwrap();
            tracker.add_observation(*child);
        }
        let last = trace.last().unwrap().0;
        let tracker = self.node_trackers.get_mut(&last).unwrap();
        tracker.add_observation(NodeAddress::SINGLETON);
        self.overall_tracker.add_observation(last);
    }

    /// The prior for the node. This is the posterior after training, but before testing.
    pub fn node_tracker(&self, na: NodeAddress) -> Option<&DirichletTracker> {
        self.node_trackers.get(&na)
    }

    /// The overall prior. This is the posterior after training, but before testing.
    pub fn overall_tracker(&self) -> &DirichletTracker {
        &self.overall_tracker
    }

    fn remove_trace(&mut self, trace: &[(NodeAddress, f32)]) {
        let parent_address_iter = trace.iter().map(|(ca, _)| ca);
        let mut child_address_iter = trace.iter().map(|(ca, _)| ca);
        child_address_iter.next();
        for (parent, child) in parent_address_iter.zip(child_address_iter) {
            let tracker = self.node_trackers.get_mut(parent).unwrap();
            tracker.remove_observation(*child);
        }
        let last = trace.last().unwrap().0;
        let tracker = self.node_trackers.get_mut(&last).unwrap();
        tracker.remove_observation(NodeAddress::SINGLETON);
        self.overall_tracker.remove_observation(last);
    }

    /// Adds an element to the trace
    pub fn add_path(&mut self, trace: Vec<(NodeAddress, f32)>) {
        self.add_trace(&trace);
        self.sequence_count += 1;
        if self.window_size != 0 {
            self.sequence_queue.push_back(trace);

            if self.sequence_queue.len() > self.window_size {
                let oldest = self.sequence_queue.pop_front().unwrap();
                self.remove_trace(&oldest);
            }
        }
    }

    /// The length of the sequence
    pub fn sequence_len(&self) -> usize {
        if self.sequence_queue.is_empty() {
            self.sequence_count
        } else {
            self.sequence_queue.len()
        }
    }

    fn stats(&self, vals: Vec<(NodeAddress, f64)>) -> CovertreeTrackerStats {
        let mut max = f64::MIN;
        let mut min = f64::MAX;
        let mut nz_count = 0;
        let mut moment1_nz = 0.0;
        let mut moment2_nz = 0.0;
        vals.iter().for_each(|(_address, kl)| {
            if *kl > 1.0e-10 {
                moment1_nz += kl;
                moment2_nz += kl * kl;
                if max < *kl {
                    max = *kl;
                }
                if *kl < min {
                    min = *kl;
                }

                nz_count += 1;
            }
        });
        CovertreeTrackerStats {
            max,
            min,
            nz_count,
            moment1_nz,
            moment2_nz,
            sequence_len: self.sequence_len(),
        }
    }

    /// Some stats about the KL divergences of the nodes in the tree
    pub fn nodes_kl_div_stats(&self) -> CovertreeTrackerStats {
        self.stats(self.nodes_kl_div())
    }

    /// Gives the per-node KL divergence, with the node address
    pub fn nodes_kl_div(&self) -> Vec<(NodeAddress, f64)> {
        self.node_trackers
            .iter()
            .map(|(address, tracker)| (*address, tracker.kl_div()))
            .collect()
    }

    /// The kl_div of the overall tree. The buckets are the individual partitions
    pub fn overall_kl_div(&self) -> f64 {
        self.overall_tracker.kl_div()
    }

    /// Some stats about the AIC of the nodes in the tree
    pub fn nodes_aic_stats(&self) -> CovertreeTrackerStats {
        self.stats(self.nodes_aic())
    }

    /// Gives the per-node aic, with the node address
    pub fn nodes_aic(&self) -> Vec<(NodeAddress, f64)> {
        self.node_trackers
            .iter()
            .map(|(address, tracker)| (*address, tracker.marginal_aic()))
            .collect()
    }

    /// The aic of the overall tree. The buckets are the individual partitions
    pub fn marginal_aic(&self) -> f64 {
        self.overall_tracker.marginal_aic()
    }

    /// Some stats about the marginal log likelihood of the nodes in the tree
    pub fn nodes_mll_stats(&self) -> CovertreeTrackerStats {
        self.stats(self.nodes_mll())
    }

    /// Gives the per-node marginal log likelihood, with the node address
    pub fn nodes_mll(&self) -> Vec<(NodeAddress, f64)> {
        self.node_trackers
            .iter()
            .map(|(address, tracker)| (*address, -tracker.mll()))
            .collect()
    }

    /// The marginal log likelihood of the overall tree. The buckets are the individual partitions
    pub fn overall_mll(&self) -> f64 {
        self.overall_tracker.mll()
    }
}

/// Tracks the non-zero KL div (all KL divergences above 1e-10)
#[derive(Debug, Serialize, Deserialize)]
pub struct CovertreeTrackerStats {
    /// The maximum non-zero KL divergence
    pub max: f64,
    /// The minimum non-zero KL divergence
    pub min: f64,
    /// The number of nodes that have a non-zero divergence
    pub nz_count: u64,
    /// The first moment, use this with the `nz_count` to get the mean
    pub moment1_nz: f64,
    /// The second moment, use this with the `nz_count` and first moment to get the variance
    pub moment2_nz: f64,
    /// The number of sequence elements that went into calculating this stat. This is not the total length
    /// We can drop old sequence elements
    pub sequence_len: usize,
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::covertree::tests::build_basic_tree;
    use crate::plugins::discrete::prelude::*;

    #[test]
    fn dirichlet_tree_probs_test() {
        let mut tree = build_basic_tree();
        tree.add_plugin::<GokoDirichlet>(GokoDirichlet::default());
        let mut tracker = BayesCovertree::new(0, &tree.reader());

        let mut unvisited_nodes = vec![tree.root_address];
        let reader = tree.reader();
        println!("Root Address: {}, ln_prob: {}", tree.root_address, 0.0);
        while let Some(addr) = unvisited_nodes.pop() {
            let ln_probs = reader
                .get_node_plugin_and::<Dirichlet, _, _>(addr, |p| p.param_vec())
                .unwrap();

            for (child_addr, child_prob) in ln_probs {
                let na = NodeAddress::from(child_addr);
                println!(
                    "Address: {:?} --- raw {} --- ln_prob: {}, parent: {}",
                    na, child_addr, child_prob, addr
                );
                if !na.singleton() {
                    unvisited_nodes.push(na);
                }
            }
        }

        assert_approx_eq!(tracker.overall_kl_div(), 0.0);
        tracker.add_path(vec![
            ((-1, 4).into(), 0.0),
            ((-2, 1).into(), 0.0),
            ((-6, 1).into(), 0.0),
        ]);
        println!("KL Div: {}", tracker.overall_kl_div());
        tracker.add_path(vec![((-1, 4).into(), 0.0)]);
        println!("KL Div: {}", tracker.overall_kl_div());
    }

    #[test]
    fn dirichlet_tree_append_test() {
        let mut tree = build_basic_tree();
        tree.add_plugin::<GokoDirichlet>(GokoDirichlet::default());
        let mut tracker = BayesCovertree::new(0, &tree.reader());
        assert_approx_eq!(tracker.overall_kl_div(), 0.0);
        tracker.add_path(vec![
            ((-1, 4).into(), 0.0),
            ((-2, 1).into(), 0.0),
            ((-6, 1).into(), 0.0),
        ]);
        println!("KL Div: {}", tracker.overall_kl_div());
        tracker.add_path(vec![((-1, 4).into(), 0.0)]);
        println!("KL Div: {}", tracker.overall_kl_div());
    }
}
