//! # Plugin System
//!
//! To implement a plugin you need to write 2 components one which implements `NodePlugin` and another that implements `TreePlugin`.
//! Finally you need to create an object that implements the parent trait that glues the two objects together.
//!
//! The `NodePlugin` is attached to each node. It is created by the `node_component` function in the parent trait when the plugin is
//! attached to the tree. It can access the `TreePlugin` component, and the tree. These are created recursively, so you can access the
//! plugin for the child nodes.
//!
//! None of this is parallelized. We need to move to Tokio to take advantage of the async computation there to || it.

use crate::node::CoverNode;
use crate::tree::CoverTreeReader;
use crate::*;
use std::fmt::Debug;
use type_map::concurrent::TypeMap;

pub mod distributions;

/*
mod sequence_kl;
pub mod utils {
    //! Utilities to use the plugins
    pub use super::sequence_kl::*;
}
*/

/// Mockup for the plugin interface attached to the node. These are meant to be functions that Grandma uses to maintain the plugin.
pub trait NodePlugin<D: PointCloud>: Send + Sync + Debug {
    /// This is currently non-functional, thinking about how to efficiently use this.
    fn update(&mut self, my_node: &CoverNode<D>, my_tree: &CoverTreeReader<D>);
}

/// Mockup for the plugin parameters attached to the base of the tree.  
pub trait TreePlugin<D: PointCloud>: Send + Sync + Debug {
    /// This is currently non-functional, thinking about how to efficiently use this.
    fn update(&mut self, my_tree: &CoverTreeReader<D>);
}

/// Parent trait that make this all work. Ideally this should be included in the `TreePlugin` but rust doesn't like it.
pub trait GrandmaPlugin<D: PointCloud> {
    /// The node component of this plugin, these are attached to each node recursively when the plug in is attached to the tree.
    type NodeComponent: NodePlugin<D> + Clone + 'static;
    /// This should largely be an object that manages the parameters of the plugin.
    type TreeComponent: TreePlugin<D> + Clone + 'static;
    /// The function that actually builds the node components.
    fn node_component(
        parameters: &Self::TreeComponent,
        my_node: &CoverNode<D>,
        my_tree: &CoverTreeReader<D>,
    ) -> Self::NodeComponent;
}

pub(crate) type NodePluginSet = TypeMap;
pub(crate) type TreePluginSet = TypeMap;

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tree::tests::build_basic_tree;

    #[derive(Debug, Clone)]
    struct DumbNode1 {
        id: u32,
        pi: PointIndex,
        cover_count: usize,
    }

    impl<D: PointCloud> NodePlugin<D> for DumbNode1 {
        fn update(&mut self, _my_node: &CoverNode<D>, _my_tree: &CoverTreeReader<D>) {
            self.id += 1;
        }
    }

    #[derive(Debug, Clone)]
    struct DumbTree1 {
        id: u32,
    }

    impl<D: PointCloud> TreePlugin<D> for DumbTree1 {
        fn update(&mut self, _my_tree: &CoverTreeReader<D>) {
            self.id += 1;
        }
    }

    #[derive(Debug)]
    struct DumbGrandma1 {}

    impl<D: PointCloud> GrandmaPlugin<D> for DumbGrandma1 {
        type NodeComponent = DumbNode1;
        type TreeComponent = DumbTree1;
        fn node_component(
            parameters: &Self::TreeComponent,
            my_node: &CoverNode<D>,
            my_tree: &CoverTreeReader<D>,
        ) -> Self::NodeComponent {
            println!(
                "Building Dumb Plugin for {:?}",
                (my_node.scale_index(), my_node.center_index())
            );
            let cover_count = match my_node.children() {
                None => my_node.singleton_len(),
                Some((nested_scale, child_addresses)) => {
                    println!(
                        "trying to get at the nodes at {:?}",
                        (nested_scale, child_addresses)
                    );
                    let mut cover_count = my_tree
                        .get_node_plugin_and::<Self::NodeComponent, _, _>(
                            (nested_scale, *my_node.center_index()),
                            |p| p.cover_count,
                        )
                        .unwrap();
                    for ca in child_addresses {
                        cover_count += my_tree
                            .get_node_plugin_and::<Self::NodeComponent, _, _>(*ca, |p| {
                                p.cover_count
                            })
                            .unwrap();
                    }
                    cover_count
                }
            };
            DumbNode1 {
                id: parameters.id,
                pi: *my_node.center_index(),
                cover_count,
            }
        }
    }

    #[test]
    fn dumb_plugins() {
        let d = DumbTree1 { id: 1 };
        let mut tree = build_basic_tree();
        tree.add_plugin::<DumbGrandma1>(d);
        println!("{:?}", tree.reader().len());
        for (si, layer) in tree.reader().layers() {
            println!("Scale Index: {:?}", si);
            layer.for_each_node(|pi, n| {
                println!("Node: {:?}", n);
                n.get_plugin_and::<DumbNode1, _, _>(|dp| {
                    println!("DumbNodes: {:?}", dp);
                    assert_eq!(*pi, dp.pi);
                });
            });
        }
    }
}
