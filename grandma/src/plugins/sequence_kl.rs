//! # Computes the hierarchical  KL divergence of the last N elements feed to this sequence
//!
//!

use super::*;
use std::fmt;
use std::f64;
use std::collections::{HashMap, VecDeque};
use rand::{thread_rng, Rng};

use crossbeam_channel::{unbounded, Receiver, Sender};
use crate::*;

pub trait InsertDistributionTracker<M: Metric>: Debug {

    fn add_trace(&mut self, trace: Vec<NodeAddress>);

    fn running_pdfs(&self) -> &HashMap<NodeAddress, BucketProbs>;

    fn tree_reader(&self) -> &CoverTreeReader<M>;
    fn sequence_len(&self) -> usize;
    fn current_stats(&self) -> KLDivergenceStats;

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

impl<M: Metric> fmt::Debug for BucketHKLDivergence<M> {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "PointCloud {{ sequence: {:?}, length: {}, running_pdfs: {:#?}}}",
            self.sequence, self.length, self.running_pdfs,
        )
    }
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
    fn sequence_len(&self) -> usize {
        self.sequence.len()
    }
    fn current_stats(&self) -> KLDivergenceStats {
        let mut stats = KLDivergenceStats::new(self.sequence_len());
        stats.add_tracker(self);
        stats
    }
}

/// Computes a frequentist KL divergence calculation on each node the sequence touches.
pub struct SGDHKLDivergence<M: Metric> {
    learning_rate: f64,
    momentum: f64,
    sequence_len: usize,
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
            sequence_len: 0,
        }
    }
}

impl<M: Metric> InsertDistributionTracker<M> for SGDHKLDivergence<M> {
    /// Adds an element to the trace
    fn add_trace(&mut self, trace: Vec<NodeAddress>) {
        self.sequence_len += 1;
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
    fn sequence_len(&self) -> usize {
        self.sequence_len
    }
    fn current_stats(&self) -> KLDivergenceStats {
        let mut stats = KLDivergenceStats::new(self.sequence_len());
        stats.add_tracker(self);
        stats
    }
}

pub struct KLDivergenceStats {
    pub moment1_max: f64,
    pub moment2_max: f64,
    pub moment1_min: f64,
    pub moment2_min: f64,
    pub moment1_nz_count: usize,
    pub moment2_nz_count: usize,
    pub moment1_mean: f64,
    pub moment2_mean: f64,
    pub moment1_nz: f64,
    pub moment2_nz: f64,
    pub sequence_len: usize,
    pub nz_total_count: usize,
    pub sequence_count: usize,
}

impl fmt::Debug for KLDivergenceStats {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "KLDivergenceStats {{mean_max : {}, var_max : {}, mean_min : {}, var_min : {}, mean_nz_count : {}, var_nz_count : {}, mean_mean : {}, var_mean : {}, mean_nz : {}, var_nz : {}, nz_total_count: {}, sequence_count: {}, sequence_len: {}}}",
            self.mean_max(),
            self.var_max(),
            self.mean_min(),
            self.var_min(),
            self.mean_nz_count(),
            self.var_nz_count(),
            self.mean_mean(),
            self.var_mean(),
            self.mean_nz(),
            self.var_nz(),
            self.nz_total_count,
            self.sequence_count,
            self.sequence_len,
        )
    }
}

impl KLDivergenceStats {
    pub fn new(sequence_len:usize) -> Self {
        KLDivergenceStats {
            moment1_max: 0.0,
            moment2_max: 0.0,
            moment1_min: 0.0,
            moment2_min: 0.0,
            moment1_nz_count: 0,
            moment2_nz_count: 0,
            moment1_mean: 0.0,
            moment2_mean: 0.0,
            moment1_nz: 0.0,
            moment2_nz: 0.0,
            nz_total_count: 0,
            sequence_count: 0,
            sequence_len,
        }
    }

    pub fn mean_max(&self) -> f64 {
        self.moment1_max/(self.sequence_count as f64)
    }
    pub fn var_max(&self) -> f64 {
        self.moment2_max/(self.sequence_count as f64) - self.mean_max()*self.mean_max()
    }
    pub fn mean_min(&self) -> f64 {
        self.moment1_min/(self.sequence_count as f64)
    }
    pub fn var_min(&self) -> f64 {
        self.moment2_min/(self.sequence_count as f64) - self.mean_min()*self.mean_min()
    }
    pub fn mean_nz_count(&self) -> f64 {
        (self.moment1_nz_count as f64)/(self.sequence_count as f64)
    }
    pub fn var_nz_count(&self) -> f64 {
        (self.moment2_nz_count as f64)/(self.sequence_count as f64) - self.mean_nz_count()*self.mean_nz_count()
    }
    pub fn mean_mean(&self) -> f64 {
        self.moment1_mean/(self.sequence_count as f64)
    }
    pub fn var_mean(&self) -> f64 {
        self.moment2_mean/(self.sequence_count as f64) - self.mean_mean()*self.mean_mean()
    }
    pub fn mean_nz(&self) -> f64 {
        self.moment1_nz/(self.nz_total_count as f64)
    }
    pub fn var_nz(&self) -> f64 {
        self.moment2_nz/(self.nz_total_count as f64) - self.mean_nz()*self.mean_nz()
    }

    pub fn add_tracker<T,M>(&mut self,tracker:&T)
    where
        T: InsertDistributionTracker<M>,
        M: Metric,
    {   
        assert!(self.sequence_len == tracker.sequence_len(), "Attempted to add a tracker to a results of the wrong lenght");
        let mut max = f64::MIN;
        let mut min = f64::MAX;
        let mut sequence_len = 0.0;
        let mut nz_count = 0;
        let mut count = 0;
        let mut moment1 = 0.0;
        let mut moment2 = 0.0;
        tracker.running_pdfs()
            .iter()
            .for_each(|(address, sequence_pdf)| {
                let kl = tracker
                    .tree_reader()
                    .get_node_plugin_and::<BucketProbs, _, _>(*address, |p| {
                        sequence_pdf.kl_divergence(p)
                    })
                    .unwrap()
                    .unwrap_or(0.0);
                if kl != 0.0 {
                    moment1 += kl;
                    moment2 += kl*kl;
                    if max < kl {
                        max = kl;
                    }
                    if kl < min {
                        min = kl;
                    }
                    
                    nz_count += 1;
                }
            });
        let mean = moment1/(nz_count as f64);
        self.moment1_max += max;
        self.moment2_max += max*max;
        self.moment1_min += min;
        self.moment2_min += min*min;
        self.moment1_nz_count += nz_count;
        self.moment2_nz_count += nz_count*nz_count;
        self.moment1_mean += mean;
        self.moment2_mean += mean*mean;
        self.nz_total_count += nz_count;
        self.moment1_nz += moment1;
        self.moment2_nz += moment2;
        self.sequence_count += 1;
    }
}

pub struct KLDivergenceTrainer<M:Metric> {
    sequence_len: usize,
    sequence_count: usize,
    learning_rate: f64,
    momentum: f64,
    reader: CoverTreeReader<M>,
}

impl<M: Metric> KLDivergenceTrainer<M> {
    pub fn new(reader: CoverTreeReader<M>) -> KLDivergenceTrainer<M> {
        KLDivergenceTrainer {
            sequence_len: 1000,
            sequence_count: 200,
            learning_rate: 0.001,
            momentum: 0.8,
            reader,
        }
    }

    pub fn set_sequence_len(&mut self, sequence_len: usize) {
        self.sequence_len = sequence_len;
    }
    pub fn set_sequence_count(&mut self, sequence_count: usize) {
        self.sequence_count = sequence_count;
    }
    pub fn set_learning_rate(&mut self, learning_rate: f64) {
        self.learning_rate = learning_rate;
    }
    pub fn set_momentum(&mut self, momentum: f64) {
        self.momentum = momentum;
    }

    pub fn train(&self) -> GrandmaResult<Vec<KLDivergenceStats>> {
        /*
        let chunk_size = 10;
        let (results_sender, results_receiver): (
            Sender<KLDivergenceStats>,
            Receiver<KLDivergenceStats>,
        ) = unbounded();
        */
        let mut results: Vec<KLDivergenceStats> = (0..self.sequence_len).map(|i| KLDivergenceStats::new(i+1)).collect();
        let point_cloud = self.reader.point_cloud();
        for _ in 0..self.sequence_count {
            let mut tracker = SGDHKLDivergence::new(self.learning_rate,self.momentum,self.reader.clone());
            for i in 0..self.sequence_len {
                let mut rng = thread_rng();
                let query_point = point_cloud.get_point(rng.gen_range(0, point_cloud.len()) as u64)?;
                tracker.add_trace(self.reader.dry_insert(query_point)?.iter().map(|(_,a)|*a).collect());
                results[i].add_tracker(&tracker);
            }
        }

        Ok(results)   
    }
}