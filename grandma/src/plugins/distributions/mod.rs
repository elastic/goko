//! # Probability Distributions Plugins
//!
//! This module containes plugins that simulate probability distributions on the nodes.
//! It also has trackers used to see when queries and sequences are out of distribution.

use super::*;
use crate::*;
use std::collections::HashMap;
use std::fmt::Debug;

mod diag_gaussian;
pub use diag_gaussian::*;

mod categorical;
pub use categorical::*;

mod dirichlet;
pub use dirichlet::*;

///
pub trait DiscreteDistribution: Clone + 'static {
    /// Pass none if you want to test for a singleton, returns 0 if
    fn ln_prob(&self, child: Option<&NodeAddress>) -> Option<f64>;

    /// Computes the KL divergence of two bucket probs.
    /// KL(self || other)
    /// Returns None if the support of the self is not a subset of the support of the other, or the calculation is undefined.
    fn kl_divergence(&self, other: &Self) -> Option<f64>;
}

///
pub trait ContinousDistribution: Clone + 'static {
    /// Pass none if you want to test for a singleton, returns 0 if
    fn ln_prob(&self, point: &[f32]) -> Option<f64>;

    /// Computes the KL divergence of two bucket probs.
    /// KL(self || other)
    /// Returns None if the support of the self is not a subset of the support of the other, or the calculation is undefined.
    fn kl_divergence(&self, other: &Self) -> Option<f64>;
}

///
pub trait DiscreteBayesianDistribution: DiscreteDistribution + Clone + 'static {
    /// Adds an observation to the distribution.
    /// This currently shifts the underlying parameters of the distribution rather than be tracked.
    fn add_observation(&mut self, loc: Option<NodeAddress>);
}

///
pub trait ContinousBayesianDistribution: ContinousDistribution + Clone + 'static {
    /// Adds an observation to the distribution.
    /// This currently shifts the underlying parameters of the distribution rather than be tracked.
    fn add_observation(&mut self, point: &[f32]);
}

/// Tracks the KL divergence for a given distribution.
pub trait DiscreteBayesianSequenceTracker<M: Metric>: Debug {
    /// The. underlying distribution that this is tracking.
    type Distribution: DiscreteBayesianDistribution + NodePlugin<M> + 'static;

    /// Adds a dry insert.
    fn add_dry_insert(&mut self, trace: Vec<(f32, NodeAddress)>);
    /// The current distributions that a dry insert touched.
    fn running_distributions(&self) -> &HashMap<NodeAddress, Self::Distribution>;
    /// Helper function, each sequence tracker should carry it's own reader.
    fn tree_reader(&self) -> &CoverTreeReader<M>;
    /// The length of the sequence
    fn sequence_len(&self) -> usize;
    /// A set of stats for the sequence that are helpful.
    fn current_stats(&self) -> KLDivergenceStats {
        let mut max = f64::MIN;
        let mut min = f64::MAX;
        let mut nz_count = 0;
        let mut moment1_nz = 0.0;
        let mut moment2_nz = 0.0;
        // For computing the fracta dimensions
        let mut layer_totals: Vec<u64> = vec![0; self.tree_reader().len()];
        let mut layer_node_counts = vec![Vec::<usize>::new(); self.tree_reader().len()];
        let parameters = self.tree_reader().parameters();
        self.running_distributions()
            .iter()
            .for_each(|(address, sequence_pdf)| {
                let kl = self
                    .tree_reader()
                    .get_node_plugin_and::<Self::Distribution, _, _>(*address, |p| {
                        p.kl_divergence(sequence_pdf).unwrap()
                    })
                    .unwrap();
                if kl > 1.0e-10 {
                    layer_totals[parameters.internal_index(address.0)] += 1;
                    layer_node_counts[parameters.internal_index(address.0)].push(
                        self.tree_reader()
                            .get_node_and(*address, |n| n.cover_count)
                            .unwrap(),
                    );

                    moment1_nz += kl;
                    moment2_nz += kl * kl;
                    if max < kl {
                        max = kl;
                    }
                    if kl < min {
                        min = kl;
                    }

                    nz_count += 1;
                }
            });
        let weighted_layer_totals: Vec<f32> = layer_node_counts.iter().map(|counts| {
            let max: f32 = *counts.iter().max().unwrap_or(&1) as f32;
            counts.iter().fold(0.0, |a,c| a + (*c as f32)/max)
        }).collect();
        KLDivergenceStats {
            max,
            min,
            nz_count,
            moment1_nz,
            moment2_nz,
            sequence_len: self.sequence_len() as u64,
            layer_totals,
            weighted_layer_totals,
        }
    }

    /// Gives the per-node KL divergence, with the node address
    fn all_node_kl(&self) -> Vec<(f64, NodeAddress)> {
        self.running_distributions()
            .iter()
            .map(|(address, sequence_pdf)| {
                let kl = self
                    .tree_reader()
                    .get_node_plugin_and::<Self::Distribution, _, _>(*address, |p| {
                        p.kl_divergence(sequence_pdf).unwrap()
                    })
                    .unwrap();
                (kl, *address)
            })
            .collect()
    }
}

/// Tracks the non-zero (all KL divergences above 1e-10)
#[derive(Debug)]
pub struct KLDivergenceStats {
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
    pub sequence_len: u64,
    /// The number of nodes per layer this sequence touches
    pub layer_totals: Vec<u64>,
    /// The number of nodes per layer this sequence touches
    pub weighted_layer_totals: Vec<f32>,
}
