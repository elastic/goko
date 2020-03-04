
use crate::tree::CoverTreeReader;
use crate::node::CoverNode;
use crate::*;

use anymap::AnyMap;
use std::fmt::Debug;
use std::any::TypeId;
use std::collections::HashMap;

pub trait NodePlugin<M>: Send + Sync + Debug 
where M: Metric {
    fn update(&mut self, my_node: &CoverNode, my_tree:&CoverTreeReader<M>);
}

#[derive(Debug)]
pub struct NodePluginSet<M: Metric> {
    data: HashMap<TypeId, Box<dyn NodePlugin<M>>>,
}

impl<M: Metric> NodePluginSet<M> {
    pub fn new() -> Self {
        Self {
            data: HashMap::new()
        }
    }
    /// Retrieve the value stored in the map for the type `T`, if it exists.
    pub fn get<'a, T: 'static + NodePlugin<M>>(&'a self) -> Option<& T> {
        self.data.get(&TypeId::of::<T>()).map(|any| any.as_ref())
    }

    /// Retrieve a mutable reference to the value stored in the map for the type `T`, if it exists.
    pub fn get_mut<'a, T: 'static + NodePlugin<M>>(&'a mut self) -> Option<&mut T> {
        self.data.get_mut(&TypeId::of::<T>()).map(|any| any.as_mut())
    }

    /// Set the value contained in the map for the type `T`.
    /// This will override any previous value stored.
    pub fn insert<T: 'static + NodePlugin<M>>(&mut self, value: T) {
        self.data.insert(TypeId::of::<T>(), Box::new(value));
    }

    /// Remove the value for the type `T` if it existed.
    pub fn remove<T: 'static + NodePlugin<M>>(&mut self) {
        self.data.remove(&TypeId::of::<T>());
    }
}


pub trait TreePlugin<M: Metric>: Send + Sync + Debug {
    type NodeComponent: NodePlugin<M>;
    fn new_node_comp(&self) -> Self::NodeComponent;
    fn update(&mut self, my_tree:&CoverTreeReader<M>);
}

#[derive(Debug)]
pub struct TreePluginSet<M: Metric> {
    data: HashMap<TypeId, Box<dyn TreePlugin<M>>>,
}

impl<M: Metric> TreePluginSet<M> {
    pub fn new() -> Self {
        Self {
            data: HashMap::new()
        }
    }
    /// Retrieve the value stored in the map for the type `T`, if it exists.
    pub fn get<'a, T: 'static + TreePlugin<M>>(&'a self) -> Option<&T> {
        self.data.get(&TypeId::of::<T>()).map(|any| any.as_ref())
    }

    /// Retrieve a mutable reference to the value stored in the map for the type `T`, if it exists.
    pub fn get_mut<'a, T: 'static + TreePlugin<M>>(&'a mut self) -> Option<&mut T> {
        self.data.get_mut(&TypeId::of::<T>()).map(|any| any.as_mut())
    }

    /// Set the value contained in the map for the type `T`.
    /// This will override any previous value stored.
    pub fn insert<T: 'static + TreePlugin<M>>(&mut self, value: T) {
        self.data.insert(TypeId::of::<T>(), Box::new(value));
    }

    /// Remove the value for the type `T` if it existed.
    pub fn remove<T: 'static + TreePlugin<M>>(&mut self) {
        self.data.remove(&TypeId::of::<T>());
    }
}



#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tree::tests::build_basic_tree;

    #[derive(Debug)]
    struct DumbNode1 {
        id: u32
    }

    impl<M: Metric> NodePlugin<M> for DumbNode1 {
        fn update(&mut self, _my_node: &CoverNode, _my_tree: &CoverTreeReader<M>) {
            self.id += 1;
        }
    }

    #[derive(Debug)]
    struct DumbTree1 {
        id: u32,
    }

    impl<M: Metric> TreePlugin<M> for DumbTree1 {
        type NodeComponent = DumbNode1;
        fn new_node_comp(&self) -> DumbNode1 {
            DumbNode1 {
                id: 0
            }
        }
        fn update(&mut self, my_tree: &CoverTreeReader<M>) {
            self.id += 1;
        }
    }


    #[test]
    fn dumb_plugins() {
        let mut d = DumbTree1 {id: 0};
        let mut tree = self.build_basic_tree();
        tree.add_plugin(d);
    }
}