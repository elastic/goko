//! Plugin for labels and metadata

use super::*;
use crate::covertree::node::CoverNode;
use crate::covertree::CoverTreeReader;
//use pointcloud::*;
use std::sync::Arc;

/// Wrapper around the summary found in the point cloud
#[derive(Debug, Default)]
pub struct NodeLabelSummary<T: Summary + Clone> {
    /// The summary object, refenced counted to eliminate duplicates
    pub summary: Arc<SummaryCounter<T>>,
}

impl<T: Summary + Clone> Clone for NodeLabelSummary<T> {
    fn clone(&self) -> Self {
        NodeLabelSummary {
            summary: Arc::clone(&self.summary),
        }
    }
}

impl<D: PointCloud> NodePlugin<D> for NodeLabelSummary<D::LabelSummary> {}

/// Plug in that allows for summaries of labels to be attached to
#[derive(Debug, Clone, Default)]
pub struct LabelSummaryPlugin {}

impl<D: PointCloud> GokoPlugin<D> for LabelSummaryPlugin {
    type NodeComponent = NodeLabelSummary<D::LabelSummary>;
    fn node_component(
        _parameters: &Self,
        my_node: &CoverNode<D>,
        my_tree: &CoverTreeReader<D>,
    ) -> Option<Self::NodeComponent> {
        let mut bucket = my_tree
            .parameters()
            .point_cloud
            .label_summary(my_node.singletons())
            .unwrap();
        // If we're a routing node then grab the childen's values
        if let Some(child_addresses) = my_node.children() {
            for ca in child_addresses {
                my_tree.get_node_plugin_and::<Self::NodeComponent, _, _>(*ca, |p| {
                    bucket.combine(p.summary.as_ref())
                });
            }
        } else {
            bucket.add(
                my_tree
                    .parameters()
                    .point_cloud
                    .label(my_node.center_index()),
            );
        }
        Some(NodeLabelSummary {
            summary: Arc::new(bucket),
        })
    }
}

/// Wrapper around the summary found in the point cloud
#[derive(Debug, Default)]
pub struct NodeMetaSummary<T: Summary + Clone> {
    /// The summary object, refenced counted to eliminate duplicates
    pub summary: Arc<SummaryCounter<T>>,
}

impl<T: Summary + Clone> Clone for NodeMetaSummary<T> {
    fn clone(&self) -> Self {
        NodeMetaSummary {
            summary: Arc::clone(&self.summary),
        }
    }
}

impl<D: PointCloud> NodePlugin<D> for NodeMetaSummary<D::MetaSummary> {}

/// Plug in that allows for summaries of Metas to be attached to
#[derive(Debug, Clone, Default)]
pub struct MetaSummaryPlugin {}

impl<D: PointCloud> GokoPlugin<D> for MetaSummaryPlugin {
    type NodeComponent = NodeMetaSummary<D::MetaSummary>;
    fn node_component(
        _parameters: &Self,
        my_node: &CoverNode<D>,
        my_tree: &CoverTreeReader<D>,
    ) -> Option<Self::NodeComponent> {
        let mut bucket = my_tree
            .parameters()
            .point_cloud
            .metasummary(my_node.singletons())
            .unwrap();
        // If we're a routing node then grab the childen's values
        if let Some(child_addresses) = my_node.children() {
            for ca in child_addresses {
                my_tree.get_node_plugin_and::<Self::NodeComponent, _, _>(*ca, |p| {
                    bucket.combine(p.summary.as_ref())
                });
            }
        } else {
            bucket.add(
                my_tree
                    .parameters()
                    .point_cloud
                    .metadata(my_node.center_index()),
            );
        }
        Some(NodeMetaSummary {
            summary: Arc::new(bucket),
        })
    }
}
