//! Plugin for labels and metadata

use super::*;
use crate::covertree::node::CoverNode;
use crate::covertree::CoverTreeReader;
//use pointcloud::*;
use std::sync::Arc;

/// Wrapper around the summary found in the point cloud
#[derive(Debug, Default)]
pub struct NodeLabelSummary<T: Summary> {
    /// The summary object, refenced counted to eliminate duplicates
    pub summary: Arc<SummaryCounter<T>>,
}

impl<T: Summary> Clone for NodeLabelSummary<T> {
    fn clone(&self) -> Self {
        NodeLabelSummary {
            summary: Arc::clone(&self.summary),
        }
    }
}

impl<D: PointCloud + LabeledCloud> NodePlugin<D> for NodeLabelSummary<D::LabelSummary> {}

///
#[derive(Debug, Clone, Default)]
pub struct TreeLabelSummary {}

impl<D: PointCloud + LabeledCloud> TreePlugin<D> for TreeLabelSummary {}

/// Plug in that allows for summaries of labels to be attached to
#[derive(Debug, Clone, Default)]
pub struct LabelSummaryPlugin {}

impl<D: PointCloud + LabeledCloud> GokoPlugin<D> for LabelSummaryPlugin {
    type NodeComponent = NodeLabelSummary<D::LabelSummary>;
    type TreeComponent = TreeLabelSummary;
    fn node_component(
        _parameters: &Self::TreeComponent,
        my_node: &CoverNode<D>,
        my_tree: &CoverTreeReader<D>,
    ) -> Option<Self::NodeComponent> {
        let mut bucket = my_tree
            .parameters()
            .point_cloud
            .label_summary(my_node.singletons())
            .unwrap();
        // If we're a routing node then grab the childen's values
        if let Some((nested_scale, child_addresses)) = my_node.children() {
            my_tree.get_node_plugin_and::<Self::NodeComponent, _, _>(
                (nested_scale, *my_node.center_index()),
                |p| bucket.combine(p.summary.as_ref()),
            );

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
                    .label(*my_node.center_index()),
            );
        }
        Some(NodeLabelSummary {
            summary: Arc::new(bucket),
        })
    }
}

/// Wrapper around the summary found in the point cloud
#[derive(Debug, Default)]
pub struct NodeMetaSummary<T: Summary> {
    /// The summary object, refenced counted to eliminate duplicates
    pub summary: Arc<SummaryCounter<T>>,
}

impl<T: Summary> Clone for NodeMetaSummary<T> {
    fn clone(&self) -> Self {
        NodeMetaSummary {
            summary: Arc::clone(&self.summary),
        }
    }
}

impl<D: PointCloud + MetaCloud> NodePlugin<D> for NodeMetaSummary<D::MetaSummary> {}

///
#[derive(Debug, Clone, Default)]
pub struct TreeMetaSummary {}

impl<D: PointCloud + MetaCloud> TreePlugin<D> for TreeMetaSummary {}

/// Plug in that allows for summaries of Metas to be attached to
#[derive(Debug, Clone, Default)]
pub struct MetaSummaryPlugin {}

impl<D: PointCloud + MetaCloud> GokoPlugin<D> for MetaSummaryPlugin {
    type NodeComponent = NodeMetaSummary<D::MetaSummary>;
    type TreeComponent = TreeMetaSummary;
    fn node_component(
        _parameters: &Self::TreeComponent,
        my_node: &CoverNode<D>,
        my_tree: &CoverTreeReader<D>,
    ) -> Option<Self::NodeComponent> {
        let mut bucket = my_tree
            .parameters()
            .point_cloud
            .metasummary(my_node.singletons())
            .unwrap();
        // If we're a routing node then grab the childen's values
        if let Some((nested_scale, child_addresses)) = my_node.children() {
            my_tree.get_node_plugin_and::<Self::NodeComponent, _, _>(
                (nested_scale, *my_node.center_index()),
                |p| bucket.combine(p.summary.as_ref()),
            );

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
                    .metadata(*my_node.center_index()),
            );
        }
        Some(NodeMetaSummary {
            summary: Arc::new(bucket),
        })
    }
}
