use crate::node::CoverNode;
use crate::tree::CoverTreeReader;
use crate::*;
use anymap::SyncAnyMap as SyncMap;

use std::any::TypeId;
use std::collections::HashMap;
use std::fmt::Debug;

pub trait NodePlugin<M: Metric>: Send + Sync + Debug {
    fn update(&mut self, my_node: &CoverNode<M>, my_tree: &CoverTreeReader<M>);
}

//type SyncMap = Map<Any + Send + Sync>;

pub type NodePluginSet = SyncMap;

pub trait TreePlugin<M: Metric>: Send + Sync + Debug {
    fn update(&mut self, my_tree: &CoverTreeReader<M>);
}
pub type TreePluginSet = SyncMap;

pub trait GrandmaPlugin<M: Metric> {
    type NodeComponent: NodePlugin<M> + Clone + 'static;
    type TreeComponent: TreePlugin<M> + Clone + 'static;
    fn node_component(
        parameters: &Self::TreeComponent,
        my_node: &CoverNode<M>,
        my_tree: &CoverTreeReader<M>,
    ) -> Self::NodeComponent;
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tree::tests::build_basic_tree;

    #[derive(Debug, Clone)]
    struct DumbNode1 {
        id: u32,
        pi: PointIndex,
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
            DumbNode1 {
                id: parameters.id,
                pi: *my_node.center_index(),
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
