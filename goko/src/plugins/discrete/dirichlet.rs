//! # Dirichlet probability
//!
//! We know that the users are quering based on what they want to know about.
//! This has some geometric structure, especially for attackers. We have some
//! prior knowlege about the queries, they should function similarly to the training set.
//! Sections of data that are highly populated should have a higher likelyhood of being
//! queried.
//!
//! This plugin lets us simulate the unknown distribution of the queries of a user in a
//! bayesian way. There may be more applications of this idea, but defending against
//! attackers has been proven.

use crate::covertree::node::CoverNode;
use crate::covertree::CoverTreeReader;
use crate::plugins::*;
pub use stats_goko::discrete::{Dirichlet, DiscreteData};

/// Simple probability density function for where things go by count
///

impl<D: PointCloud> NodePlugin<D> for Dirichlet {}

/// Stores the log probabilities for each node in the tree.
///
/// This is the probability that when you sample from the tree you end up at a particular node.
#[derive(Debug, Clone, Default)]
pub struct GokoDirichlet {
    // probability that you'd pass thru this node.
//pub cond_ln_probs: HashMap<NodeAddress,f64>,
}

/// Parent trait that make this all work. Ideally this should be included in the `TreePlugin` but rust doesn't like it.
impl<D: PointCloud> GokoPlugin<D> for GokoDirichlet {
    type NodeComponent = Dirichlet;
    fn node_component(
        _parameters: &Self,
        my_node: &CoverNode<D>,
        my_tree: &CoverTreeReader<D>,
    ) -> Option<Self::NodeComponent> {
        let mut bucket = DiscreteData::new();

        // If we're a routing node then grab the childen's values
        if let Some(child_addresses) = my_node.children() {
            for ca in child_addresses {
                my_tree
                    .get_node_and(*ca, |n| {
                        bucket.add_pop(*ca, n.coverage_count() as f64 + 1.0);
                    })
                    .unwrap();
            }
            bucket.add_pop(
                NodeAddress::SINGLETON,
                my_node.singletons_len() as f64 + 1.0,
            );
        } else {
            bucket.add_pop(
                NodeAddress::SINGLETON,
                my_node.singletons_len() as f64 + 1.0,
            );
        }
        Some(bucket.into())
    }
}
