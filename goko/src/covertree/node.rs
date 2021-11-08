/*
* Licensed to Elasticsearch B.V. under one or more contributor
* license agreements. See the NOTICE file distributed with
* this work for additional information regarding copyright
* ownership. Elasticsearch B.V. licenses this file to you under
* the Apache License, Version 2.0 (the "License"); you may
* not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*  http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/

//! # The Node
//! This is the workhorse of the library. Each node
//!
use super::query_tools::{RoutingQueryHeap, SingletonQueryHeap};
use crate::errors::{GokoError, GokoResult};
use crate::plugins::{
    labels::{NodeLabelSummary, NodeMetaSummary},
    NodePlugin, NodePluginSet,
};
use crate::tree_file_format::*;
use crate::NodeAddress;
use smallvec::smallvec;
use std::ops::Deref;

use pointcloud::*;
use smallvec::SmallVec;
use std::marker::PhantomData;
use std::sync::Arc;

/// The actual cover node. The fields can be separated into three piles. The first two consist of node `address` for testing and reference
/// when working and the `radius`, `coverage_count`, and `singles_summary` for a query various properties of the node.
/// Finally we have the children and singleton pile. The singletons are saved in a `SmallVec` directly attached to the node. This saves a
/// memory redirect for the first 20 singleton children. The children are saved in a separate struct also consisting of a `SmallVec`
/// (though, this is only 10 wide before we allocate on the heap), and the scale index of the nested child.
#[derive(Debug)]
pub struct CoverNode<D: PointCloud> {
    /// Parent address
    parent_address: Option<NodeAddress>,
    /// Node address
    address: NodeAddress,
    /// Query caches
    radius: f32,
    coverage_count: usize,
    /// Children
    children: Option<SmallVec<[NodeAddress; 2]>>,
    singles_indexes: SmallVec<[usize; 5]>,
    plugins: NodePluginSet,
    metic: PhantomData<D>,
}

impl<D: PointCloud> Clone for CoverNode<D> {
    fn clone(&self) -> Self {
        Self {
            parent_address: self.parent_address,
            address: self.address,
            radius: self.radius,
            coverage_count: self.coverage_count,
            children: self.children.clone(),
            singles_indexes: self.singles_indexes.clone(),
            plugins: NodePluginSet::new(),
            metic: PhantomData,
        }
    }
}

impl<D: PointCloud> CoverNode<D> {
    /// Creates a new blank node
    pub fn new(parent_address: Option<NodeAddress>, address: NodeAddress) -> CoverNode<D> {
        CoverNode {
            parent_address,
            address,
            radius: 0.0,
            coverage_count: 1,
            children: None,
            singles_indexes: SmallVec::new(),
            plugins: NodePluginSet::new(),
            metic: PhantomData,
        }
    }

    /// If the node has a summary attached, this returns the summary.
    pub fn label_summary(&self) -> Option<Arc<SummaryCounter<D::LabelSummary>>> {
        self.plugins
            .get::<NodeLabelSummary<D::LabelSummary>>()
            .map(|c| Arc::clone(&c.summary))
    }

    /// If the node has a summary attached, this returns the summary.
    pub fn metasummary(&self) -> Option<Arc<SummaryCounter<D::MetaSummary>>> {
        self.plugins
            .get::<NodeMetaSummary<D::MetaSummary>>()
            .map(|c| Arc::clone(&c.summary))
    }

    /// Verifies that this is a leaf by checking there's no nested child
    pub fn is_leaf(&self) -> bool {
        self.children.is_none()
    }

    /// This is currently inconsistent on inserts to children of this node
    pub fn radius(&self) -> f32 {
        self.radius
    }

    /// Number of decendents of this node
    pub fn coverage_count(&self) -> usize {
        self.coverage_count
    }

    /// Reads the contents of a plugin, due to the nature of the plugin map we have to access it with a
    /// closure.
    pub fn get_plugin_and<T: Send + Sync + 'static, F, S>(&self, transform_fn: F) -> Option<S>
    where
        F: FnOnce(&T) -> S,
    {
        self.plugins.get::<T>().map(transform_fn)
    }

    /// Removes all children and returns them to us.
    pub(crate) fn remove_children(&mut self) -> Option<SmallVec<[NodeAddress; 2]>> {
        self.children.take()
    }

    /// The number of singleton points attached to the node
    pub fn singletons_len(&self) -> usize {
        self.singles_indexes.len()
    }

    ///
    pub fn singletons(&self) -> &[usize] {
        &self.singles_indexes
    }

    ///
    pub fn address(&self) -> NodeAddress {
        self.address
    }

    ///
    pub fn center_index(&self) -> usize {
        self.address.point_index()
    }

    ///
    pub fn scale_index(&self) -> i32 {
        self.address.scale_index()
    }

    ///
    pub fn parent_address(&self) -> Option<NodeAddress> {
        self.parent_address
    }

    ///
    pub fn children_len(&self) -> usize {
        match &self.children {
            Some(children) => children.len(),
            None => 0,
        }
    }

    /// If the node is not a leaf this unpacks the child struct to a more publicly consumable format.
    pub fn children(&self) -> Option<&[NodeAddress]> {
        self.children.as_ref().map(|c| &c[..])
    }

    /// Performs the `singleton_knn` and `child_knn` with a provided query heap. If you have the distance
    /// from the query point to this you can pass it to save a distance calculation.
    pub fn knn<
        P: Deref<Target = D::Point> + Send + Sync,
        T: SingletonQueryHeap + RoutingQueryHeap,
    >(
        &self,
        dist_to_center: Option<f32>,
        point: &P,
        point_cloud: &D,
        query_heap: &mut T,
    ) -> GokoResult<()> {
        self.singleton_knn(point, point_cloud, query_heap)?;

        let dist_to_center = dist_to_center
            .unwrap_or(point_cloud.distances_to_point(point, &[self.address.point_index()])?[0]);
        self.child_knn(Some(dist_to_center), point, point_cloud, query_heap)?;

        if self.children.is_none() {
            query_heap.push_outliers(&[self.address.point_index()], &[dist_to_center]);
        }
        Ok(())
    }

    /// Performs a brute force knn against just the singleton children with a provided query heap.
    pub fn singleton_knn<P: Deref<Target = D::Point> + Send + Sync, T: SingletonQueryHeap>(
        &self,
        point: &P,
        point_cloud: &D,
        query_heap: &mut T,
    ) -> GokoResult<()> {
        let distances = point_cloud.distances_to_point(point, &self.singles_indexes[..])?;
        query_heap.push_outliers(&self.singles_indexes[..], &distances[..]);
        Ok(())
    }

    /// Performs a brute force knn against the children of the node with a provided query heap. Does nothing if this is a leaf node.
    /// If you have the distance from the query point to this you can pass it to save a distance calculation.
    pub fn child_knn<P: Deref<Target = D::Point> + Send + Sync, T: RoutingQueryHeap>(
        &self,
        dist_to_center: Option<f32>,
        point: &P,
        point_cloud: &D,
        query_heap: &mut T,
    ) -> GokoResult<()> {
        let dist_to_center = dist_to_center
            .unwrap_or(point_cloud.distances_to_point(point, &[self.address.point_index()])?[0]);

        if let Some(children) = &self.children {
            query_heap.push_nodes(&[children[0]], &[dist_to_center], None);
            let children_indexes: Vec<usize> =
                children[1..].iter().map(|na| na.point_index()).collect();
            let distances = point_cloud.distances_to_point(point, &children_indexes[..])?;
            query_heap.push_nodes(&children[1..], &distances, Some(self.address));
        }
        Ok(())
    }

    /// Gives the closest routing node to the query point.
    pub fn nearest_covering_child<P: Deref<Target = D::Point> + Send + Sync>(
        &self,
        scale_base: f32,
        dist_to_center: f32,
        point: &P,
        point_cloud: &D,
    ) -> GokoResult<Option<(NodeAddress, f32)>> {
        if let Some(children) = &self.children {
            let children_indexes: Vec<usize> =
                children[1..].iter().map(|na| na.point_index()).collect();
            let distances = point_cloud.distances_to_point(point, &children_indexes[..])?;
            let (min_index, min_dist) = distances
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                .unwrap_or((0, &std::f32::MAX));
            if dist_to_center < *min_dist {
                if dist_to_center < scale_base.powi(children[0].scale_index()) {
                    Ok(Some((children[0], dist_to_center)))
                } else {
                    Ok(None)
                }
            } else if *min_dist < scale_base.powi(children[min_index + 1].scale_index()) {
                Ok(Some((children[min_index + 1], *min_dist)))
            } else {
                Ok(None)
            }
        } else {
            Ok(None)
        }
    }

    /// Gives the child that the point would be inserted into if the
    /// point just happened to never be picked as a center. This is the first child node that covers
    /// the point.
    pub fn first_covering_child<P: Deref<Target = D::Point> + Send + Sync>(
        &self,
        scale_base: f32,
        dist_to_center: f32,
        point: &P,
        point_cloud: &D,
    ) -> GokoResult<Option<(NodeAddress, f32)>> {
        if let Some(children) = &self.children {
            if dist_to_center < scale_base.powi(children[0].scale_index()) {
                return Ok(Some((children[0], dist_to_center)));
            }
            let children_indexes: Vec<usize> =
                children[1..].iter().map(|na| na.point_index()).collect();
            let distances = point_cloud.distances_to_point(point, &children_indexes[..])?;
            for (ca, d) in children[1..].iter().zip(distances) {
                if d < scale_base.powi(ca.scale_index()) {
                    return Ok(Some((*ca, d)));
                }
            }
        }
        Ok(None)
    }

    /// Add a nested child and converts the node from a leaf to a routing node.
    /// Throws an error if the node is already a routing node with a nested node.
    pub(crate) fn insert_nested_child(
        &mut self,
        scale_index: i32,
        coverage: usize,
    ) -> GokoResult<()> {
        assert!(
            coverage > 0,
            "Panic: attempting to add an empty nested child."
        );
        self.coverage_count += coverage - 1;
        if self.children.is_some() {
            Err(GokoError::DoubleNest)
        } else {
            self.children = Some(smallvec![NodeAddress::from((
                scale_index,
                self.address.point_index()
            ))]);
            Ok(())
        }
    }

    /// Inserts a routing child into the node. Make sure the child node is also in the tree or you get a dangling reference
    pub(crate) fn insert_child(&mut self, address: NodeAddress, coverage: usize) -> GokoResult<()> {
        self.coverage_count += coverage;
        if let Some(children) = &mut self.children {
            children.push(address);
            Ok(())
        } else {
            Err(GokoError::InsertBeforeNest)
        }
    }

    /// Inserts a `vec` of singleton children into the node.
    pub(crate) fn insert_singletons(&mut self, addresses: Vec<usize>) {
        self.coverage_count += addresses.len();
        self.singles_indexes.extend(addresses);
    }
    /// Inserts a single singleton child into the node.
    pub(crate) fn insert_singleton(&mut self, pi: usize) {
        self.coverage_count += 1;
        self.singles_indexes.push(pi);
    }

    /// Inserts a single singleton child into the node.
    pub(crate) fn insert_plugin<T: NodePlugin<D> + 'static>(&mut self, plugin: T) {
        self.plugins.insert(plugin);
    }

    /// Updates the radius
    pub(crate) fn set_radius(&mut self, radius: f32) {
        self.radius = radius;
    }

    pub(crate) fn load(node_proto: &NodeProto) -> CoverNode<D> {
        let singles_indexes = node_proto
            .outlier_point_indexes
            .iter()
            .map(|i| *i as usize)
            .collect();
        let radius = node_proto.get_radius();
        let address = NodeAddress::from((
            node_proto.get_scale_index(),
            node_proto.get_center_index() as usize,
        ));
        let parent_scale_index = node_proto.get_parent_scale_index();
        let parent_center_index = node_proto.get_parent_center_index();
        let parent_address =
            if parent_scale_index == std::i32::MIN && parent_center_index == std::u64::MAX {
                None
            } else {
                Some(NodeAddress::from((
                    parent_scale_index,
                    parent_center_index as usize,
                )))
            };
        let coverage_count = node_proto.get_coverage_count() as usize;
        let children = if node_proto.get_is_leaf() {
            None
        } else {
            Some(
                node_proto
                    .get_children_scale_indexes()
                    .iter()
                    .zip(node_proto.get_children_point_indexes())
                    .map(|(si, pi)| (*si as i32, *pi as usize).into())
                    .collect(),
            )
        };
        CoverNode {
            parent_address,
            address,
            radius,
            coverage_count,
            children,
            singles_indexes,
            plugins: NodePluginSet::new(),
            metic: PhantomData,
        }
    }

    pub(crate) fn save(&self) -> NodeProto {
        let mut proto = NodeProto::new();
        proto.set_coverage_count(self.coverage_count as u64);
        proto.set_scale_index(self.address.scale_index());
        proto.set_center_index(self.address.point_index() as u64);

        match self.parent_address {
            Some(parent_address) => {
                proto.set_parent_scale_index(parent_address.scale_index());
                proto.set_parent_center_index(parent_address.point_index() as u64);
            }
            None => {
                proto.set_parent_scale_index(std::i32::MIN);
                proto.set_parent_center_index(std::u64::MAX);
            }
        }

        proto.set_radius(self.radius);
        proto.set_outlier_point_indexes(self.singles_indexes.iter().map(|pi| *pi as u64).collect());

        match &self.children {
            Some(children) => {
                proto.set_is_leaf(false);
                proto.set_children_scale_indexes(
                    children.iter().map(|na| na.scale_index()).collect(),
                );
                proto.set_children_point_indexes(
                    children.iter().map(|na| na.point_index() as u64).collect(),
                );
            }
            None => proto.set_is_leaf(true),
        }
        proto
    }

    /*
    /// Brute force verifies that the children are separated by at least the scale provided.
    /// The scale provided should be b^(s-1) where s is this node's scale index.
    pub fn check_seperation(&self, scale: f32, point_cloud: &D) -> GokoResult<bool> {
        let mut nodes = self.singles_indexes.clone();
        nodes.push(self.address.1);
        if let Some(children) = &self.children {
            nodes.extend(children.addresses.iter().map(|(_si, pi)| *pi));
        }
        let adj = point_cloud.adjacency_matrix(&nodes)?;
        Ok(scale > adj.min())
    }
    */
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    use crate::covertree::tests::build_mnist_tree;
    use crate::query_tools::knn_query_heap::tests::clone_unvisited_nodes;
    use crate::query_tools::query_items::QueryAddress;
    use crate::query_tools::KnnQueryHeap;

    fn create_test_node<D: PointCloud>() -> CoverNode<D> {
        let children = Some(smallvec![
            (-4, 0).into(),
            (-4, 1).into(),
            (-4, 2).into(),
            (-4, 3).into()
        ]);

        CoverNode {
            parent_address: None,
            address: (0, 0).into(),
            radius: 1.0,
            coverage_count: 8,
            children,
            singles_indexes: smallvec![4, 5, 6],
            plugins: NodePluginSet::new(),
            metic: PhantomData,
        }
    }

    fn create_test_leaf_node<D: PointCloud>() -> CoverNode<D> {
        CoverNode {
            parent_address: Some((1, 0).into()),
            address: (0, 0).into(),
            radius: 1.0,
            coverage_count: 8,
            children: None,
            singles_indexes: smallvec![1, 2, 3, 4, 5, 6],
            plugins: NodePluginSet::new(),
            metic: PhantomData,
        }
    }

    #[test]
    fn knn_node_children_mixed() {
        // Tests the mixed uppacking
        let data = vec![0.0, 0.49, 0.48, 0.5, 0.1, 0.2, 0.3];
        let labels = vec![0, 0, 0, 0, 1, 1, 1];
        let point_cloud = DefaultLabeledCloud::<L2>::new_simple(data, 1, labels);

        let test_node = create_test_node();
        let mut heap = KnnQueryHeap::new(5, 2.0);
        let point = [0.494f32];
        test_node
            .knn(None, &&point[..], &point_cloud, &mut heap)
            .unwrap();
        println!("{:?}", heap);
        println!("There shoud be 4 node addresses on the heap here");
        assert!(heap.node_len() == 4);
        println!("There shoud be only 3 singleton indexes on the heap");
        assert!(heap.len() == 5);
        let results = heap.unpack();
        println!("There should be 5 results, {:?}", results);
        assert!(results.len() == 5);
        println!("The first result should be 1 but is {:?}", results[0].1);
        assert!(results[0].0 == 1);
        println!("The first result should be 3 but is {:?}", results[1].1);
        assert!(results[1].0 == 3);
    }

    #[test]
    fn knn_node_children_only() {
        let data = vec![0.0, 0.49, 0.48, 0.5, 0.1, 0.2, 0.3];
        let labels = vec![0, 0, 0, 0, 1, 1, 1];
        let point_cloud = DefaultLabeledCloud::<L2>::new_simple(data, 1, labels);

        let test_node = create_test_node();
        let mut heap = KnnQueryHeap::new(5, 2.0);
        let point = [0.494f32];
        test_node
            .knn(None, &&point[..], &point_cloud, &mut heap)
            .unwrap();
        println!("{:?}", heap);
        println!("There shoud be 4 node addresses on the heap here");
        assert!(heap.node_len() == 4);
        println!("There shoud be only 3 singleton indexes on the heap");
        assert!(heap.len() == 5);
        let results = heap.unpack();
        println!("There should be 5 results, {:?}", results);
        assert!(results.len() == 5);
        println!("The first result should be 1 but is {:?}", results[0].1);
        assert!((results[0].0) == 1);
        println!("The first result should be 3 but is {:?}", results[1].1);
        assert!((results[1].0) == 3);
    }

    #[test]
    fn knn_node_leaf() {
        let data = vec![0.0, 0.49, 0.48, 0.5, 0.1, 0.2, 0.3];
        let labels = vec![0, 0, 0, 0, 1, 1, 1];
        let point_cloud = DefaultLabeledCloud::<L2>::new_simple(data, 1, labels);

        let test_node = create_test_leaf_node();
        let mut heap = KnnQueryHeap::new(5, 2.0);
        let point = [0.494f32];
        test_node
            .knn(None, &point.as_ref(), &point_cloud, &mut heap)
            .unwrap();
        println!("{:?}", heap);
        println!("There shoudn't be any node addresses on the heap here");
        assert!(heap.node_len() == 0);
        println!("There shoud be only 2 singleton indexes on the heap");
        assert!(heap.len() == 5);
        let results = heap.unpack();
        println!("There should be 5 results");
        assert!(results.len() == 5);
        println!("The first result should be 1 but is {:?}", results[0].1);
        assert!(results[0].0 == 1);
        println!("The first result should be 3 but is {:?}", results[1].1);
        assert!(results[1].0 == 3);
    }

    fn brute_test_knn_node(
        node: &CoverNode<DefaultLabeledCloud<L2>>,
        point_cloud: &DefaultLabeledCloud<L2>,
    ) -> bool {
        let zeros: Vec<f32> = vec![0.0; 784];
        let mut heap = KnnQueryHeap::new(10000, 1.3);
        let dist_to_center = point_cloud
            .distances_to_point(&zeros.as_ref(), &[node.address.point_index()])
            .unwrap()[0];

        node.knn(
            Some(dist_to_center),
            &zeros.as_ref(),
            &point_cloud,
            &mut heap,
        )
        .unwrap();

        // Complete KNN
        let mut all_children = Vec::from(node.singletons());
        if let Some(children) = &node.children {
            all_children.extend(children.iter().map(|na| na.point_index()));
        } else {
            all_children.push(node.center_index())
        }
        let brute_knn = point_cloud
            .distances_to_point(&zeros.as_ref(), &all_children)
            .unwrap();
        let mut brute_knn: Vec<(usize, f32)> = brute_knn
            .iter()
            .zip(all_children)
            .map(|(d, i)| (i, *d))
            .collect();
        brute_knn.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let brute_knn: Vec<usize> = brute_knn.iter().map(|(pi, _d)| *pi).collect();

        // Children KNN
        let children = &node.children().map(|c| c.to_vec()).unwrap_or(Vec::new());
        let children_indexes: Vec<usize> = children.iter().map(|na| na.point_index()).collect();

        let children_dist = point_cloud
            .distances_to_point(&zeros.as_ref(), &children_indexes)
            .unwrap();

        let mut children_range_calc: Vec<QueryAddress> = children_dist
            .iter()
            .zip(children)
            .map(|(d, na)| QueryAddress {
                min_dist: (*d - (1.3f32).powi(na.scale_index())).max(0.0),
                dist_to_center: *d,
                address: *na,
            })
            .collect();
        children_range_calc.sort();

        // Testing nearest covering child
        match (
            children_range_calc
                .iter()
                .min_by(|a, b| a.dist_to_center.partial_cmp(&b.dist_to_center).unwrap()),
            node.nearest_covering_child(1.3, dist_to_center, &zeros.as_ref(), &point_cloud)
                .unwrap(),
        ) {
            (Some(query), Some((q_a, q_d))) => {
                println!("Expected {:?}", query);
                println!("Got {:?}", (q_d, q_a));
                assert_approx_eq!(query.dist_to_center, q_d);
                assert_eq!(query.address.point_index(), q_a.point_index());
            }
            (None, None) => {}
            (Some(query), None) => {
                assert!(query.min_dist > 0.0);
            }
            _ => {
                assert!(false, "nearest_covering_child is broken");
            }
        }

        let children_range_calc: Vec<NodeAddress> =
            children_range_calc.iter().map(|a| a.address).collect();

        let heap_range: Vec<NodeAddress> = clone_unvisited_nodes(&heap)
            .iter()
            .map(|(a, _d)| *a)
            .collect();
        let heap_knn: Vec<usize> = heap.unpack().iter().map(|(pi, _d)| *pi).collect();

        let mut correct = true;
        if correct {
            correct = heap_knn == brute_knn;
        }
        if correct {
            correct = heap_range == children_range_calc;
        }
        if !correct {
            println!("=============");
            println!("Node: {:?}", node);
            println!("====");
            println!("Heap Range Calc: {:?}", heap_range);
            println!("Brute Range Calc: {:?}", children_range_calc);
            println!("Heap Knn: {:?}", heap_knn);
            println!("Brute Knn: {:?}", brute_knn);
            println!("=============");
        }
        correct
    }

    #[test]
    fn nearest_covering_child_sanity() {
        let data = vec![0.0, 0.49, 0.48, 0.5, 0.1, 0.2, 0.3];
        let labels = vec![0, 0, 0, 0, 1, 1, 1];
        let point_cloud = DefaultLabeledCloud::<L2>::new_simple(data, 1, labels);

        let test_node = create_test_node();
        let point = [0.494f32];
        let (na, d) = test_node
            .nearest_covering_child(2.0, 0.494, &&point[..], &point_cloud)
            .unwrap()
            .unwrap();
        assert_eq!(na, (-4, 1).into());
        assert_approx_eq!(d, 0.004);
    }

    #[test]
    fn mnist_knn_node_on_level() {
        if env::var("TRAVIS_RUST_VERSION").is_err() {
            let tree = build_mnist_tree();
            let reader = tree.reader();
            println!("Testing Root");
            reader
                .get_node_and(reader.root_address(), |n| {
                    brute_test_knn_node(n, reader.point_cloud())
                })
                .unwrap();

            let layer = reader.layer(reader.root_address().scale_index() - 3);
            println!("Testing 3 layers below root, with {} nodes", layer.len());
            // Allowed 3 errors.
            let mut errors = 0;
            layer.for_each_node(|_, n| {
                if !brute_test_knn_node(n, reader.point_cloud()) {
                    errors += 1;
                }
            });
            assert!(errors < 3);
        }
    }

    #[test]
    fn save_load_root_node() {
        let node = create_test_node::<DefaultLabeledCloud<L2>>();
        let proto = node.save();
        let reconstructed_node = CoverNode::<DefaultLabeledCloud<L2>>::load(&proto);

        assert_eq!(
            reconstructed_node
                .parent_address
                .map(|n| <Option<(i32, usize)>>::from(n)),
            None
        );
        assert_eq!(
            <Option<(i32, usize)>>::from(reconstructed_node.address),
            Some((0, 0))
        );
        assert_eq!(reconstructed_node.radius, 1.0);
        assert_eq!(reconstructed_node.coverage_count, 8);
        assert_eq!(&reconstructed_node.singles_indexes[..], &[4, 5, 6]);

        let reconstructed_children = reconstructed_node.children.unwrap();
        assert_eq!(
            &reconstructed_children[..],
            &[
                (-4, 0).into(),
                (-4, 1).into(),
                (-4, 2).into(),
                (-4, 3).into()
            ]
        );
    }

    #[test]
    fn save_load_leaf_node() {
        let node = create_test_leaf_node::<DefaultLabeledCloud<L2>>();
        let proto = node.save();
        let reconstructed_node = CoverNode::<DefaultLabeledCloud<L2>>::load(&proto);

        assert_eq!(
            reconstructed_node.parent_address.map(|n| n.into()),
            Some(Some((1, 0)))
        );
        assert_eq!(
            <Option<(i32, usize)>>::from(reconstructed_node.address),
            Some((0, 0))
        );
        assert_eq!(reconstructed_node.radius, 1.0);
        assert_eq!(reconstructed_node.coverage_count, 8);
        assert_eq!(&reconstructed_node.singles_indexes[..], &[1, 2, 3, 4, 5, 6]);
        assert!(reconstructed_node.children.is_none());
    }
}
