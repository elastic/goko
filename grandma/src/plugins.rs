use crate::node::CoverNode;
use crate::tree::CoverTreeReader;
use crate::*;
use anymap::Map;
use anymap::any::{UncheckedAnyExt};
use std::any::Any;

use std::any::TypeId;
use std::collections::HashMap;
use std::fmt::Debug;

pub trait NodePlugin<M:Metric>: Send + Sync + Debug {
    fn update(&mut self, my_node: &CoverNode<M>, my_tree: &CoverTreeReader<M>);
}

type SyncMap = Map<Any + Send + Sync>;

pub type NodePluginSet = SyncMap;

pub trait TreePlugin<M: Metric>: Send + Sync + Debug {
    fn update(&mut self, my_tree: &CoverTreeReader<M>);
}
pub type TreePluginSet = SyncMap;


pub trait GrandmaPlugin<M: Metric> {
    type NodeComponent: NodePlugin<M> + Clone;
    type TreeComponent: TreePlugin<M> + Clone;
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

    #[derive(Debug,Clone)]
    struct DumbNode1 {
        id: u32,
    }

    impl<M: Metric> NodePlugin<M> for DumbNode1 {
        fn update(&mut self, _my_node: &CoverNode<M>, _my_tree: &CoverTreeReader<M>) {
            self.id += 1;
        }
    }

    #[derive(Debug,Clone)]
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
                id: parameters.id
            }
        }
    }

    #[test]
    fn dumb_plugins() {
        let mut d = DumbTree1 { id: 1 };
        let mut tree = build_basic_tree();
        tree.add_plugin::<DumbGrandma1>(d);


    }
}
