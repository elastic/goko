//! # Probability Distributions Plugins
//!
//! This module containes plugins that simulate probability distributions on the nodes.
//! It also has trackers used to see when queries and sequences are out of distribution.

use super::*;
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
    fn ln_prob(&self, point: &PointRef) -> Option<f64>;

    /// Computes the KL divergence of two bucket probs.
    /// KL(self || other)
    /// Returns None if the support of the self is not a subset of the support of the other, or the calculation is undefined.
    fn kl_divergence(&self, other: &Self) -> Option<f64>;
}

///
pub trait DiscreteBayesianDistribution: DiscreteDistribution + Clone + 'static {
    /// The distribution to which this is a conjugate prior
    type Evidence: 'static + Debug;
    /// Adds an observation to the distribution.
    /// This currently shifts the underlying parameters of the distribution rather than be tracked.
    fn add_observation(&mut self, loc: Option<NodeAddress>);
    /// Adds several observations
    fn add_evidence(&mut self, evidence: &Self::Evidence);
    /// Computes the KL divergence between the prior and the posterior, with the given evidence
    fn posterior_kl_divergence(&self, other: &Self::Evidence) -> Option<f64>;
}

///
pub trait ContinousBayesianDistribution: ContinousDistribution + Clone + 'static {
    /// Adds an observation to the distribution.
    /// This currently shifts the underlying parameters of the distribution rather than be tracked.
    fn add_observation(&mut self, point: &PointRef);
}

/// Tracks the KL divergence for a given distribution.
pub trait DiscreteBayesianSequenceTracker<D: PointCloud>: Debug {
    /// The. underlying distribution that this is tracking.
    type Distribution: DiscreteBayesianDistribution + NodePlugin<D> + 'static;

    /// Adds a dry insert.
    fn add_path(&mut self, trace: Vec<(f32, NodeAddress)>);
    /// The current distributions that a dry insert touched.
    fn running_evidence(&self) -> &HashMap<NodeAddress, <<Self as plugins::distributions::DiscreteBayesianSequenceTracker<D>>::Distribution as DiscreteBayesianDistribution>::Evidence>;
    
    /// Gives the per-node KL divergence, with the node address
    fn all_node_kl(&self) -> Vec<(f64, NodeAddress)> {
        self.running_evidence()
            .iter()
            .map(|(address, sequence_pdf)| {
                let kl = self
                    .tree_reader()
                    .get_node_plugin_and::<Self::Distribution, _, _>(*address, |p| {
                        p.posterior_kl_divergence(sequence_pdf).unwrap()
                    })
                    .unwrap();
                (kl, *address)
            })
            .collect()
    }

    /// Helper function, each sequence tracker should carry it's own reader.
    fn tree_reader(&self) -> &CoverTreeReader<D>;
    /// The length of the sequence
    fn sequence_len(&self) -> usize;

    /// A set of stats for the sequence that are helpful.
    fn kl_div_stats(&self) -> KLDivergenceStats {
        let mut max = f64::MIN;
        let mut min = f64::MAX;
        let mut nz_count = 0;
        let mut moment1_nz = 0.0;
        let mut moment2_nz = 0.0;
        self.all_node_kl()
            .iter()
            .for_each(|(kl, _address)| {
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
        KLDivergenceStats {
            max,
            min,
            nz_count,
            moment1_nz,
            moment2_nz,
            sequence_len: self.sequence_len(),
        }
    }

    /// A set of stats for the sequence that are helpful.
    fn fractal_dim_stats(&self) -> FractalDimStats {
        let mut layer_totals: Vec<u64> = vec![0; self.tree_reader().len()];
        let mut layer_node_counts = vec![Vec::<usize>::new(); self.tree_reader().len()];
        let parameters = self.tree_reader().parameters();
        self.all_node_kl()
            .iter()
            .for_each(|(_kl, address)| {
                layer_totals[parameters.internal_index(address.0)] += 1;
                layer_node_counts[parameters.internal_index(address.0)].push(
                    self.tree_reader()
                        .get_node_and(*address, |n| n.cover_count())
                        .unwrap(),
                );
            });
        let weighted_layer_totals: Vec<f32> = layer_node_counts
            .iter()
            .map(|counts| {
                let max: f32 = *counts.iter().max().unwrap_or(&1) as f32;
                counts.iter().fold(0.0, |a, c| a + (*c as f32) / max)
            })
            .collect();
        FractalDimStats {
            sequence_len: self.sequence_len(),
            layer_totals,
            weighted_layer_totals,
        }
    }    
}

/// Tracks the non-zero KL div (all KL divergences above 1e-10)
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
    pub sequence_len: usize,
}

/// Stats that let you compute the fractal dim of the query dataset wrt the base covertree
pub struct FractalDimStats {
    /// The number of sequence elements that went into calculating this stat. This is not the total lenght
    /// We can drop old sequence elements
    pub sequence_len: usize,
    /// The number of nodes per layer this sequence touches
    pub layer_totals: Vec<u64>,
    /// The number of nodes per layer this sequence touches
    pub weighted_layer_totals: Vec<f32>,
}


/// Tracks the non-zero (all KL divergences above 1e-10)
#[derive(Debug, Default)]
pub struct KLDivergenceBaselineStats {
    /// The maximum non-zero KL divergence
    pub max: (f64,f64),
    /// The minimum non-zero KL divergence
    pub min: (f64,f64),
    /// The number of nodes that have a non-zero divergence
    pub nz_count: (f64,f64),
    /// The first moment, use this with the `nz_count` to get the mean
    pub moment1_nz: (f64,f64),
    /// The second moment, use this with the `nz_count` and first moment to get the variance
    pub moment2_nz: (f64,f64),
}

impl KLDivergenceBaselineStats {
    pub(crate) fn add(&mut self, stats:&KLDivergenceStats) {
        self.max.0 += stats.max;
        self.max.1 += stats.max*stats.max;
        self.min.0 += stats.min;
        self.min.1 += stats.min*stats.min;
        self.nz_count.0 += stats.nz_count as f64;
        self.nz_count.1 += (stats.nz_count*stats.nz_count) as f64;
        self.moment1_nz.0 += stats.moment1_nz;
        self.moment1_nz.1 += stats.moment1_nz*stats.moment1_nz;
        self.moment2_nz.0 += stats.moment2_nz;
        self.moment2_nz.1 += stats.moment2_nz*stats.moment2_nz;
    }

    fn to_mean_var(&self, count: f64) -> KLDivergenceBaselineStats {
        let max_mean = self.max.0/count;
        let min_mean = self.min.0/count;
        let nz_count_mean = self.nz_count.0/count;
        let moment1_nz_mean = self.moment1_nz.0/count;
        let moment2_nz_mean = self.moment2_nz.0/count;

        let max_var = self.max.1/count - max_mean*max_mean;
        let min_var = self.min.1/count - min_mean*min_mean;
        let nz_count_var = self.nz_count.1/count - nz_count_mean*nz_count_mean;
        let moment1_nz_var = self.moment1_nz.1/count - moment1_nz_mean*moment1_nz_mean;
        let moment2_nz_var = self.moment2_nz.1/count - moment2_nz_mean*moment2_nz_mean;

        KLDivergenceBaselineStats {
            max: (max_mean, max_var),
            min: (min_mean, min_var),
            nz_count: (nz_count_mean, nz_count_var),
            moment1_nz: (moment1_nz_mean, moment1_nz_var),
            moment2_nz: (moment2_nz_mean, moment2_nz_var),
        }
    }

    fn interpolate(mut self, other:&Self, w: f64) -> Self {
        self.max.0 += w*(other.max.0 - self.max.0);
        self.max.1 += w*(other.max.1 - self.max.1);

        self.min.0 += w*(other.min.0 - self.min.0);
        self.min.1 += w*(other.min.1 - self.min.1);

        self.nz_count.0 += w*(other.nz_count.0 - self.nz_count.0);
        self.nz_count.1 += w*(other.nz_count.1 - self.nz_count.1);

        self.moment1_nz.0 += w*(other.moment1_nz.0 - self.moment1_nz.0);
        self.moment1_nz.1 += w*(other.moment1_nz.1 - self.moment1_nz.1);

        self.moment2_nz.0 += w*(other.moment2_nz.0 - self.moment2_nz.0);
        self.moment2_nz.1 += w*(other.moment2_nz.1 - self.moment2_nz.1);
        self
    }
}

/// Computing the KL div of each node's prior and posterior is expensive. 
pub struct KLDivergenceBaseline {
    /// The number of sequences we're have the stats from
    pub num_sequences: usize,
    /// The lenght of the sequences we have stats for. All other values are linearly interpolated between these.
    pub sequence_len: Vec<usize>,
    /// The actual stats objects. These are stored by the moments, but are returned by (mean,var)
    pub stats: Vec<KLDivergenceBaselineStats>,
}

impl KLDivergenceBaseline {
    /// Gets the stats object that stores an approximate mean and variance of the samples. 
    pub fn stats(&self,i:usize) -> KLDivergenceBaselineStats {
        match self.sequence_len.binary_search(&i) {
            Ok(index) => {
                self.stats[index].to_mean_var(self.num_sequences as f64)
            }
            Err(index) => {
                if index == 0 {
                    KLDivergenceBaselineStats::default()
                } else if index == self.sequence_len.len() {
                    let stats1 = self.stats[index-2].to_mean_var(self.num_sequences as f64);
                    let stats2 = self.stats[index-1].to_mean_var(self.num_sequences as f64);
                    let weight = ((i- self.sequence_len[index-2]) as f64)/((self.sequence_len[index-1] - self.sequence_len[index-2]) as f64);
                    stats1.interpolate(&stats2,weight)
                } else {
                    let stats1 = self.stats[index-1].to_mean_var(self.num_sequences as f64);
                    let stats2 = self.stats[index].to_mean_var(self.num_sequences as f64);
                    let weight = ((i- self.sequence_len[index-1]) as f64)/((self.sequence_len[index] - self.sequence_len[index-1]) as f64);
                    stats1.interpolate(&stats2,weight)
                }
            },
        }
    }
}
