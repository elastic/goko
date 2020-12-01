//! Plugin for labels and metadata

use super::*;
use crate::covertree::node::CoverNode;
use crate::covertree::CoverTreeReader;
//use pointcloud::*;
use std::sync::Arc;

/// Contains all points that this node covers, if the coverage is lower than the limit set in the parameters.
#[derive(Debug, Clone)]
pub struct CoverageIndexes {
    pis: Arc<Vec<usize>>,
}

impl<D: PointCloud> NodePlugin<D> for CoverageIndexes {}

impl CoverageIndexes {
    /// Returns all point indexes that the node covers
    pub fn point_indexes(&self) -> &[usize] {
        self.pis.as_ref()
    }
}

/// A plugin that helps gather all the indexes that the node covers into an array you can use.
#[derive(Debug, Clone)]
pub struct GokoCoverageIndexes {
    /// The actual limit
    pub max: usize,
}

impl GokoCoverageIndexes {
    /// Set up the plugin for restricting the number of indexes we collect into any one node
    pub fn restricted(max: usize) -> Self {
        Self { max }
    }

    /// Set up the plugin for no restrictions
    pub fn new() -> Self {
        Self { max: usize::MAX }
    }
}

impl<D: PointCloud> GokoPlugin<D> for GokoCoverageIndexes {
    type NodeComponent = CoverageIndexes;
    fn node_component(
        parameters: &Self,
        my_node: &CoverNode<D>,
        my_tree: &CoverTreeReader<D>,
    ) -> Option<Self::NodeComponent> {
        if my_node.coverage_count() < parameters.max {
            let mut indexes = my_node.singletons().to_vec();
            // If we're a routing node then grab the childen's values
            if let Some((nested_scale, child_addresses)) = my_node.children() {
                my_tree.get_node_plugin_and::<Self::NodeComponent, _, _>(
                    (nested_scale, *my_node.center_index()),
                    |p| {
                        indexes.extend(p.point_indexes());
                    },
                );
                for ca in child_addresses {
                    my_tree.get_node_plugin_and::<Self::NodeComponent, _, _>(*ca, |p| {
                        indexes.extend(p.point_indexes());
                    });
                }
            } else {
                indexes.push(*my_node.center_index());
            }
            Some(CoverageIndexes {
                pis: Arc::new(indexes),
            })
        } else {
            None
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::covertree::tests::build_basic_tree;

    #[test]
    fn coverage_sanity() {
        let mut ct = build_basic_tree();
        ct.add_plugin::<GokoCoverageIndexes>(GokoCoverageIndexes::new());
        let ct_reader = ct.reader();
        let mut untested_addresses = vec![ct_reader.root_address()];
        while let Some(addr) = untested_addresses.pop() {
            let count = ct_reader
                .get_node_plugin_and::<CoverageIndexes, _, _>(addr, |p| p.point_indexes().len())
                .unwrap();
            ct_reader.get_node_and(addr, |n| {
                assert_eq!(n.coverage_count(), count, "Node: {:?}", n)
            });

            ct_reader.get_node_children_and(addr, |covered, children| {
                untested_addresses.push(covered);
                untested_addresses.extend(children);
            });
        }
    }
}
