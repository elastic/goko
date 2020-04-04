mod bucket_bayes;
use super::*;
use crate::*;
pub use bucket_bayes::*;
use std::collections::HashMap;
use std::fmt;
use std::fmt::Debug;

pub trait BayesianDistribution: Clone + 'static {
    fn add_observation(&mut self, loc: Option<NodeAddress>);
    fn ln_prob(&self, loc: Option<NodeAddress>) -> f64;
    fn kl_divergence(&self, other: &Self) -> f64;
}

pub trait InsertDistributionTracker<M: Metric>: Debug {
    type Distribution: BayesianDistribution + NodePlugin<M>;

    fn add_trace(&mut self, trace: Vec<NodeAddress>);
    fn running_pdfs(&self) -> &HashMap<NodeAddress, Self::Distribution>;
    fn tree_reader(&self) -> &CoverTreeReader<M>;
    fn sequence_len(&self) -> usize;

    fn current_stats(&self) -> KLDivergenceStats {
        let mut max = f64::MIN;
        let mut min = f64::MAX;
        let mut nz_count = 0;
        let mut moment1_nz = 0.0;
        let mut moment2_nz = 0.0;
        self.running_pdfs()
            .iter()
            .for_each(|(address, sequence_pdf)| {
                let kl = self
                    .tree_reader()
                    .get_node_plugin_and::<Self::Distribution, _, _>(*address, |p| {
                        p.kl_divergence(sequence_pdf)
                    })
                    .unwrap();
                if kl > 1.0e-10 {
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
        KLDivergenceStats {
            max,
            min,
            nz_count,
            moment1_nz,
            moment2_nz,
            sequence_len: self.sequence_len(),
        }
    }

    /// Gives the per-node KL divergence, with the node address
    fn all_node_kl(&self) -> Vec<(f64, NodeAddress)> {
        self.running_pdfs()
            .iter()
            .map(|(address, sequence_pdf)| {
                let kl = self
                    .tree_reader()
                    .get_node_plugin_and::<Self::Distribution, _, _>(*address, |p| {
                        p.kl_divergence(sequence_pdf)
                    })
                    .unwrap();
                (kl, *address)
            })
            .collect()
    }
}

#[derive(Debug)]
pub struct KLDivergenceStats {
    pub max: f64,
    pub min: f64,
    pub nz_count: usize,
    pub moment1_nz: f64,
    pub moment2_nz: f64,
    pub sequence_len: usize,
}
