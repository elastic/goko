//! # Categorical Distribution
//!
//! Simple probability distribution that enables you to simulated the rough
//! distribution of data in the tree.

use crate::covertree::node::CoverNode;
use crate::covertree::CoverTreeReader;
use crate::plugins::*;

use stats_goko::discrete::{Categorical, DiscreteData};

impl<D: PointCloud> NodePlugin<D> for Categorical {}

/// Zero sized type that can be passed around. Equivilant to `()`
#[derive(Debug, Clone)]
pub struct GokoCategorical {}

/// Parent trait that make this all work. Ideally this should be included in the `TreePlugin` but rust doesn't like it.
impl<D: PointCloud> GokoPlugin<D> for GokoCategorical {
    type NodeComponent = Categorical;
    fn node_component(
        _parameters: &Self,
        my_node: &CoverNode<D>,
        my_tree: &CoverTreeReader<D>,
    ) -> Option<Self::NodeComponent> {
        let mut bucket = DiscreteData::new();

        // If we're a routing node then grab the childen's values
        if let Some(child_addresses) = my_node.children() {
            for ca in child_addresses {
                my_tree.get_node_and(*ca, |n| {
                    bucket.add_pop(*ca, n.coverage_count() as f64);
                });
            }
            bucket.add_pop(NodeAddress::SINGLETON, my_node.singletons_len() as f64);
        } else {
            bucket.add_pop(
                NodeAddress::SINGLETON,
                my_node.singletons_len() as f64 + 1.0,
            );
        }
        Some(bucket.into())
    }
}
