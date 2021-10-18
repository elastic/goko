//! See the paper for how this works

use crate::covertree::CoverTreeReader;
use crate::plugins::*;
use crate::NodeAddress;
use hashbrown::HashMap;

use stats_goko::discrete::{Dirichlet, DirichletTracker};

use serde::{Deserialize, Serialize};

use std::fmt;

use std::collections::VecDeque;

/// Computes a frequentist KL divergence calculation on each node the sequence touches.
pub struct BayesCategoricalTracker<D: PointCloud> {
    overall_tracker: DirichletTracker,
    node_trackers: HashMap<NodeAddress, DirichletTracker>,
    sequence_queue: VecDeque<Vec<(f32, NodeAddress)>>,
    sequence_count: usize,
    window_size: usize,
    reader: CoverTreeReader<D>,
}

impl<D: PointCloud> fmt::Debug for BayesCategoricalTracker<D> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "PointCloud {{ sequence_queue: {:?}, window_size: {}, node_trackers: {:?}}}",
            self.sequence_queue, self.window_size, self.node_trackers,
        )
    }
}

impl<D: PointCloud> BayesCategoricalTracker<D> {
    /// Creates a new blank thing with capacity `size`, input 0 for unlimited.
    pub fn new(window_size: usize, reader: CoverTreeReader<D>) -> BayesCategoricalTracker<D> {
        let total_alpha_overall = (reader.parameters().point_cloud.len() + reader.node_count()) as f64;
        BayesCategoricalTracker {
            overall_tracker: DirichletTracker::sparse(total_alpha_overall, reader.node_count()),
            node_trackers: HashMap::new(),
            sequence_queue: VecDeque::new(),
            sequence_count: 0,
            window_size,
            reader,
        }
    }

    fn add_trace_to_pdfs(&mut self, trace: &[(f32, NodeAddress)]) {
        let parent_address_iter = trace.iter().map(|(_, ca)| ca);
        let mut child_address_iter = trace.iter().map(|(_, ca)| ca);
        child_address_iter.next();
        let node_trackers = &mut self.node_trackers;
        let overall_tracker = &mut self.overall_tracker;
        let reader = &self.reader;
        for (parent, child) in parent_address_iter.zip(child_address_iter) {
            node_trackers
                .entry(*parent)
                .or_insert_with_key(|k| {
                    reader
                        .get_node_plugin_and::<Dirichlet, _, _>(*k, |p| {
                            overall_tracker.set_alpha(k.raw(), p.get_alpha(NodeAddress::SINGLETON_U64).expect("There should always be singletons."));
                            p.tracker()
                        })
                        .unwrap()
                })
                .add_observation(child.raw());
        }
        let last = trace.last().unwrap().1;

        node_trackers
            .entry(last)
            .or_insert_with_key(|k| {
                reader
                    .get_node_plugin_and::<Dirichlet, _, _>(*k, |p| {
                        overall_tracker.set_alpha(k.raw(), p.get_alpha(NodeAddress::SINGLETON_U64).expect("There should always be singletons."));
                        p.tracker()
                    })
                    .unwrap()
            })
            .add_observation(NodeAddress::SINGLETON_U64);
        overall_tracker.add_observation(last.raw());
    }

    fn remove_trace_from_pdfs(&mut self, trace: &[(f32, NodeAddress)]) {
        let parent_address_iter = trace.iter().map(|(_, ca)| ca);
        let mut child_address_iter = trace.iter().map(|(_, ca)| ca);
        child_address_iter.next();
        for (parent, child) in parent_address_iter.zip(child_address_iter) {
            let parent_evidence = self.node_trackers.get_mut(parent).unwrap();
            parent_evidence.remove_observation(child.raw());
        }
        let last = trace.last().unwrap().1;
        self.node_trackers
            .get_mut(&last)
            .unwrap()
            .remove_observation(NodeAddress::SINGLETON_U64);
        self.overall_tracker.add_observation(last.raw());
    }

    /// Adds an element to the trace
    pub fn add_path(&mut self, trace: Vec<(f32, NodeAddress)>) {
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

    /// The lenght of the sequence
    pub fn sequence_len(&self) -> usize {
        if self.sequence_queue.is_empty() {
            self.sequence_count
        } else {
            self.sequence_queue.len()
        }
    }

    fn stats(&self, vals: Vec<(f64, NodeAddress)>) -> CovertreeTrackerStats {
        let mut max = f64::MIN;
        let mut min = f64::MAX;
        let mut nz_count = 0;
        let mut moment1_nz = 0.0;
        let mut moment2_nz = 0.0;
        vals.iter().for_each(|(kl, _address)| {
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
    pub fn node_kl_div_stats(&self) -> CovertreeTrackerStats {
        self.stats(self.node_kl_div())
    }

    /// Gives the per-node KL divergence, with the node address
    pub fn node_kl_div(&self) -> Vec<(f64, NodeAddress)> {
        self.node_trackers
            .iter()
            .map(|(address, tracker)| (tracker.kl_div(), *address))
            .collect()
    }

    /// The kl_div of the overall tree. The buckets are the individual partitions
    pub fn kl_div(&self) -> f64 {
        self.overall_tracker.kl_div()
    }

    /// Some stats about the AIC of the nodes in the tree
    pub fn node_aic_stats(&self) -> CovertreeTrackerStats {
        self.stats(self.node_aic())
    }

    /// Gives the per-node aic, with the node address
    pub fn node_aic(&self) -> Vec<(f64, NodeAddress)> {
        self.node_trackers
            .iter()
            .map(|(address, tracker)| (tracker.marginal_aic(), *address))
            .collect()
    }

    /// The aic of the overall tree. The buckets are the individual partitions
    pub fn marginal_aic(&self) -> f64 {
        self.overall_tracker.marginal_aic()
    }

    /// Easy access to the cover tree read head associated to this tracker
    pub fn reader(&self) -> &CoverTreeReader<D> {
        &self.reader
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
    /// The number of sequence elements that went into calculating this stat. This is not the total lenght
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
        let mut tracker = BayesCategoricalTracker::new(0, tree.reader());

        let mut unvisited_nodes = vec![tree.root_address];
        let reader = tree.reader();
        println!("Root Address: {}, ln_prob: {}", tree.root_address, 0.0);
        while let Some(addr) = unvisited_nodes.pop() {
            let ln_probs = reader
                .get_node_plugin_and::<Dirichlet, _, _>(addr, |p| p.param_vec())
                .unwrap().unwrap();
            
            for (child_addr, child_prob) in ln_probs {
                println!(
                    "Address: {:?} --- raw {} --- ln_prob: {}, parent: {}",
                    NodeAddress::from_u64(child_addr), child_addr, child_prob, addr
                );
                if let Some(ca) = NodeAddress::from_u64(child_addr) {
                    unvisited_nodes.push(ca);
                }
            }
        }

        assert_approx_eq!(tracker.kl_div(), 0.0);
        tracker.add_path(vec![
            (0.0, (-1, 4).into()),
            (0.0, (-2, 1).into()),
            (0.0, (-6, 1).into()),
        ]);
        println!("KL Div: {}", tracker.kl_div());
        tracker.add_path(vec![(0.0, (-1, 4).into())]);
        println!("KL Div: {}", tracker.kl_div());
        
    }

    #[test]
    fn dirichlet_tree_append_test() {
        let mut tree = build_basic_tree();
        tree.add_plugin::<GokoDirichlet>(GokoDirichlet::default());
        let mut tracker = BayesCategoricalTracker::new(0, tree.reader());
        assert_approx_eq!(tracker.kl_div(), 0.0);
        tracker.add_path(vec![
            (0.0, (-1, 4).into()),
            (0.0, (-2, 1).into()),
            (0.0, (-6, 1).into()),
        ]);
        println!("KL Div: {}", tracker.kl_div());
        tracker.add_path(vec![(0.0, (-1, 4).into())]);
        println!("KL Div: {}", tracker.kl_div());
    }
}
