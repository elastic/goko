//use crossbeam_channel::unbounded;
use std::collections::HashMap;
use crate::*;

pub struct BulkInterface<D: PointCloud> {
    reader: CoverTreeReader<D>,
    final_addresses: HashMap<PointIndex,NodeAddress>,
}

impl<D: PointCloud> BulkInterface<D> {
    pub fn new(reader: CoverTreeReader<D>) -> Self {
        let mut unvisited_nodes: Vec<NodeAddress> = vec![reader.root_address()];
        let mut final_addresses = HashMap::new();
        while !unvisited_nodes.is_empty() {
            let cur_add = unvisited_nodes.pop().unwrap();
            reader.get_node_and(cur_add, |n| {
                for singleton in n.singletons() {
                    final_addresses.insert(*singleton,cur_add);
                }
                if let Some((nested_si,child_addresses)) = n.children() {
                    unvisited_nodes.extend(child_addresses);
                    unvisited_nodes.push((nested_si,cur_add.1));
                }
            }).unwrap();
        }

        BulkInterface {
            reader,
            final_addresses,
        }
    }

    pub fn known_dry_trace(&self, point_index: PointIndex) -> Option<Vec<NodeAddress>> {
        self.final_addresses.get(&point_index).map(|addr| {
            let mut path = Vec::with_capacity((self.reader.root_address().0 - addr.0) as usize);
            let mut parent = Some(*addr);
            while let Some(addr) = parent {
                path.push(addr);
                parent = self.reader.get_node_and(addr,|n| n.parent_address()).flatten();
            }
            (&mut path[..]).reverse();
            path
        })
    }
}