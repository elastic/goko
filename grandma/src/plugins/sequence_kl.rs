//! # Computes the hierarchical  KL divergence of the last N elements feed to this sequence
//!
//!

use super::*;
use std::fmt;

use std::collections::{HashMap, VecDeque};

pub trait InsertDistributionTracker<M: Metric> {
    fn add_trace(&mut self, trace: Vec<NodeAddress>);

    fn running_pdfs(&self) -> &HashMap<NodeAddress, BucketProbs>;

    fn tree_reader(&self) -> &CoverTreeReader<M>;

    fn max_node_kl(&self) -> Option<(f64, NodeAddress)> {
        let mut max_kl: Option<(f64, NodeAddress)> = None;
        for (address, sequence_pdf) in self.running_pdfs().iter() {
            let kl = self
                .tree_reader()
                .get_node_plugin_and::<BucketProbs, _, _>(*address, |p| {
                    sequence_pdf.kl_divergence(p)
                })
                .unwrap();
            if let Some(kl) = kl {
                if let Some(mkl) = max_kl.as_mut() {
                    if kl > mkl.0 {
                        *mkl = (kl, *address);
                    }
                } else {
                    max_kl = Some((kl, *address));
                }
            }
        }
        max_kl
    }

    /// Gives the per-node KL divergence, with the node address
    fn all_node_kl(&self) -> Vec<(f64, NodeAddress)> {
        self.running_pdfs()
            .iter()
            .map(|(address, sequence_pdf)| {
                let kl = self
                    .tree_reader()
                    .get_node_plugin_and::<BucketProbs, _, _>(*address, |p| {
                        sequence_pdf.kl_divergence(p)
                    })
                    .unwrap()
                    .unwrap_or(0.0);
                (kl, *address)
            })
            .collect()
    }
}

/// Computes a frequentist KL divergence calculation on each node the sequence touches.
pub struct BucketHKLDivergence<M: Metric> {
    running_pdfs: HashMap<NodeAddress, BucketProbs>,
    sequence: VecDeque<Vec<NodeAddress>>,
    length: usize,
    reader: CoverTreeReader<M>,
}

impl<M: Metric> BucketHKLDivergence<M> {
    /// Creates a new blank thing with capacity `size`, input 0 for unlimited.
    pub fn new(size: usize, reader: CoverTreeReader<M>) -> BucketHKLDivergence<M> {
        BucketHKLDivergence {
            running_pdfs: HashMap::new(),
            sequence: VecDeque::new(),
            length: size,
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
                .or_insert(BucketProbs::new())
                .add_child_pop(Some(*child), 1);
        }
        let last = trace.last().unwrap();
        self.running_pdfs
            .entry(*last)
            .or_insert(BucketProbs::new())
            .add_child_pop(None, 1);
    }

    fn remove_trace_from_pdfs(&mut self, trace: &[NodeAddress]) {
        let parent_address_iter = trace.iter();
        let mut child_address_iter = trace.iter();
        child_address_iter.next();
        for (parent, child) in parent_address_iter.zip(child_address_iter) {
            self.running_pdfs
                .entry(*parent)
                .or_insert(BucketProbs::new())
                .remove_child_pop(Some(*child), 1);
        }
        let last = trace.last().unwrap();
        self.running_pdfs
            .entry(*last)
            .or_insert(BucketProbs::new())
            .remove_child_pop(None, 1);
    }
}
impl<M: Metric> InsertDistributionTracker<M> for BucketHKLDivergence<M> {
    /// Adds an element to the trace
    fn add_trace(&mut self, trace: Vec<NodeAddress>) {
        self.add_trace_to_pdfs(&trace);
        self.sequence.push_back(trace);
        if self.sequence.len() > self.length && self.length != 0 {
            let oldest = self.sequence.pop_front().unwrap();
            self.remove_trace_from_pdfs(&oldest);
        }
    }
    fn running_pdfs(&self) -> &HashMap<NodeAddress, BucketProbs> {
        &self.running_pdfs
    }
    fn tree_reader(&self) -> &CoverTreeReader<M> {
        &self.reader
    }
}

/// Computes a frequentist KL divergence calculation on each node the sequence touches.
pub struct SGDHKLDivergence<M: Metric> {
    learning_rate: f64,
    momentum: f64,
    running_pdfs: HashMap<NodeAddress, BucketProbs>,
    reader: CoverTreeReader<M>,
}

impl<M: Metric> fmt::Debug for SGDHKLDivergence<M> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "PointCloud {{ learning_rate: {}, momentum: {}, running_pdfs: {:#?}}}",
            self.learning_rate, self.momentum, self.running_pdfs,
        )
    }
}

impl<M: Metric> SGDHKLDivergence<M> {
    pub fn new(
        learning_rate: f64,
        momentum: f64,
        reader: CoverTreeReader<M>,
    ) -> SGDHKLDivergence<M> {
        SGDHKLDivergence {
            running_pdfs: HashMap::new(),
            learning_rate,
            momentum,
            reader,
        }
    }
}

impl<M: Metric> InsertDistributionTracker<M> for SGDHKLDivergence<M> {
    /// Adds an element to the trace
    fn add_trace(&mut self, trace: Vec<NodeAddress>) {
        let parent_address_iter = trace.iter();
        let mut child_address_iter = trace.iter();
        child_address_iter.next();
        for (parent, child) in parent_address_iter.zip(child_address_iter) {
            self.running_pdfs
                .entry(*parent)
                .or_insert(
                    self.reader
                        .get_node_plugin_and::<BucketProbs, _, _>(*parent, |p| p.clone())
                        .unwrap(),
                )
                .sgd_observation(Some(child), self.learning_rate, self.momentum);
        }
        let last = trace.last().unwrap();
        self.running_pdfs
            .entry(*last)
            .or_insert(
                self.reader
                    .get_node_plugin_and::<BucketProbs, _, _>(*last, |p| p.clone())
                    .unwrap(),
            )
            .sgd_observation(None, self.learning_rate, self.momentum);
    }
    fn running_pdfs(&self) -> &HashMap<NodeAddress, BucketProbs> {
        &self.running_pdfs
    }
    fn tree_reader(&self) -> &CoverTreeReader<M> {
        &self.reader
    }
}
