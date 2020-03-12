use crate::node::CoverNode;
use crate::tree::CoverTreeReader;
use crate::*;
use anymap::SendSyncAnyMap;

use std::any::TypeId;
use std::collections::HashMap;
use std::fmt::Debug;

pub mod diag_gaussian;

pub use diag_gaussian::*;

pub trait NodePlugin<M: Metric>: Send + Sync + Debug {
    fn update(&mut self, my_node: &CoverNode<M>, my_tree: &CoverTreeReader<M>);
}

pub trait TreePlugin<M: Metric>: Send + Sync + Debug {
    fn update(&mut self, my_tree: &CoverTreeReader<M>);
}

pub trait GrandmaPlugin<M: Metric> {
    type NodeComponent: NodePlugin<M> + Clone + 'static;
    type TreeComponent: TreePlugin<M> + Clone + 'static;
    fn node_component(
        parameters: &Self::TreeComponent,
        my_node: &CoverNode<M>,
        my_tree: &CoverTreeReader<M>,
    ) -> Self::NodeComponent;
}

pub type NodePluginSet = SendSyncAnyMap;
pub type TreePluginSet = SendSyncAnyMap;

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

    impl<M: Metric> NodePlugin<M> for DumbNode1 {
        fn update(&mut self, _my_node: &CoverNode<M>, _my_tree: &CoverTreeReader<M>) {
            self.id += 1;
        }
    }

    #[derive(Debug, Clone)]
    struct DumbTree1 {
        id: u32,
    }

    impl<M: Metric> TreePlugin<M> for DumbTree1 {
        fn update(&mut self, my_tree: &CoverTreeReader<M>) {
            self.id += 1;
        }
    }

    #[derive(Debug)]
    struct DumbGrandma1 {}

    impl<M: Metric> GrandmaPlugin<M> for DumbGrandma1 {
        type NodeComponent = DumbNode1;
        type TreeComponent = DumbTree1;
        fn node_component(
            parameters: &Self::TreeComponent,
            my_node: &CoverNode<M>,
            my_tree: &CoverTreeReader<M>,
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
        let mut d = DumbTree1 { id: 1 };
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