//! # Bucket probability
//!
//! A class for handling the finite probablity distribution of the children

use super::*;

use std::collections::HashMap;

/// Simple probability density function for where things go by count
///
#[derive(Debug, Clone)]
pub struct BucketProbs {
    child_counts: HashMap<NodeAddress, usize>,
    singleton_count: usize,
    total: usize,
}

impl BucketProbs {
    /// Creates a new empty bucket probability
    pub fn new() -> BucketProbs {
        BucketProbs {
            child_counts: HashMap::new(),
            singleton_count: 0,
            total: 0,
        }
    }

    ///
    pub fn total(&self) -> usize {
        self.total
    }

    /// Adds the coverage to the key given by the child, pass none to add to the singleton pop
    pub fn add_child_pop(&mut self, child_address: Option<NodeAddress>, child_coverage: usize) {
        match child_address {
            Some(ca) => {
                *self.child_counts.entry(ca).or_insert(0) += child_coverage;
            }
            None => self.singleton_count += child_coverage,
        }
        self.total += child_coverage;
    }

    /// Adds the coverage to the key given by the child, pass none to add to the singleton pop
    pub fn remove_child_pop(&mut self, child_address: Option<NodeAddress>, child_coverage: usize) {
        match child_address {
            Some(ca) => {
                let cp = self.child_counts.entry(ca).or_insert(0);
                if *cp < child_coverage {
                    *cp = 0;
                } else {
                    *cp -= child_coverage;
                }
            }
            None => {
                if self.singleton_count < child_coverage {
                    self.singleton_count = 0;
                } else {
                    self.singleton_count -= child_coverage;
                }
            }
        }
        self.total += child_coverage;
    }

    /// Pass none if you want to test for a singleton, returns 0 if
    pub fn pdf(&self, child: Option<&NodeAddress>) -> f32 {
        if let Some(ca) = child {
            match self.child_counts.get(ca) {
                Some(child_count) => (*child_count as f32) / (self.total as f32),
                None => 0.0,
            }
        } else {
            (self.singleton_count as f32) / (self.total as f32)
        }
    }

    /// Computes the KL divergence of two bucket probs.
    /// KL(self || other)
    /// Returns None if the support of the self is not a subset of the support of the other
    pub fn kl_divergence(&self, other: &BucketProbs) -> Option<f32> {
        let mut sum: f32 = self.pdf(None) * (self.pdf(None) / other.pdf(None)).ln();
        for (k, v) in self.child_counts.iter() {
            if let Some(ov) = other.child_counts.get(k) {
                let p = (*v as f32) / (self.total as f32);
                let q = (*ov as f32) / (self.total as f32);
                sum += p * (p.ln() - q.ln());
            } else {
                return None;
            }
        }
        Some(sum)
    }
}

impl<M: Metric> NodePlugin<M> for BucketProbs {
    fn update(&mut self, _my_node: &CoverNode<M>, _my_tree: &CoverTreeReader<M>) {}
}

/// Zero sized type that can be passed around. Equivilant to `()`
#[derive(Debug, Clone)]
pub struct BucketProbsTree {}

impl<M: Metric> TreePlugin<M> for BucketProbsTree {
    fn update(&mut self, _my_tree: &CoverTreeReader<M>) {}
}

/// Zero sized type that can be passed around. Equivilant to `()`
#[derive(Debug, Clone)]
pub struct BucketProbsGrandma {}

/// Parent trait that make this all work. Ideally this should be included in the `TreePlugin` but rust doesn't like it.
impl<M: Metric> GrandmaPlugin<M> for BucketProbsGrandma {
    type NodeComponent = BucketProbs;
    type TreeComponent = BucketProbsTree;
    fn node_component(
        _parameters: &Self::TreeComponent,
        my_node: &CoverNode<M>,
        my_tree: &CoverTreeReader<M>,
    ) -> Self::NodeComponent {
        let mut bucket = BucketProbs::new();

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
            bucket.add_child_pop(None, my_node.singleton_len());
        } else {
            bucket.add_child_pop(None, my_node.singleton_len() + 1);
        }
        bucket
    }
}
