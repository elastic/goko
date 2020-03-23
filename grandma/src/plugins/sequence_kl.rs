//! # Computes the hierarchical  KL divergence of the last N elements feed to this sequence
//!
//!

use super::*;

use std::collections::{HashMap, VecDeque};

pub struct BucketHKLDivergence {
    running_pdfs: HashMap<NodeAddress, BucketProbs>,
    sequence: VecDeque<Vec<NodeAddress>>,
    length: usize,
}

impl BucketHKLDivergence {
    pub fn new(size: usize) -> BucketHKLDivergence {
        BucketHKLDivergence {
            running_pdfs: HashMap::new(),
            sequence: VecDeque::new(),
            length: size,
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

    pub fn add_trace(&mut self, trace: Vec<NodeAddress>) {
        self.add_trace_to_pdfs(&trace);
        self.sequence.push_back(trace);
        if self.sequence.len() > self.length {
            let oldest = self.sequence.pop_front().unwrap();
            self.remove_trace_from_pdfs(&oldest);
        }
    }

    pub fn max_kl<M: Metric>(
        &self,
        tree_reader: &CoverTreeReader<M>,
    ) -> Option<(f32, NodeAddress)> {
        let mut max_kl: Option<(f32, NodeAddress)> = None;
        for (address, sequence_pdf) in self.running_pdfs.iter() {
            let kl = tree_reader
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
}
