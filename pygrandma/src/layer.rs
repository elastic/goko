use pyo3::prelude::*;

use ndarray::{Array, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::PyIterProtocol;

use grandma::layer::*;
use grandma::*;
use pointcloud::*;
use std::sync::Arc;

use crate::node::*;

#[pyclass]
pub struct IterLayers {
    pub parameters: Arc<CoverTreeParameters<DefaultLabeledCloud<L2>>>,
    pub tree: Arc<CoverTreeReader<DefaultLabeledCloud<L2>>>,
    pub scale_indexes: Vec<i32>,
    pub index: usize,
}

impl std::iter::Iterator for IterLayers {
    type Item = PyGrandLayer;
    fn next(&mut self) -> Option<PyGrandLayer> {
        if self.index < self.scale_indexes.len() {
            self.index += 1;
            Some(PyGrandLayer {
                parameters: Arc::clone(&self.parameters),
                tree: Arc::clone(&self.tree),
                scale_index: self.scale_indexes[self.index - 1],
            })
        } else {
            None
        }
    }
}

#[pyproto]
impl PyIterProtocol for IterLayers {
    fn __iter__(slf: PyRefMut<Self>) -> PyResult<Py<IterLayers>> {
        Ok(slf.into())
    }
    fn __next__(mut slf: PyRefMut<Self>) -> PyResult<Option<PyGrandLayer>> {
        Ok(slf.next())
    }
}

#[pyclass]
pub struct PyGrandLayer {
    pub parameters: Arc<CoverTreeParameters<DefaultLabeledCloud<L2>>>,
    pub tree: Arc<CoverTreeReader<DefaultLabeledCloud<L2>>>,
    pub scale_index: i32,
}

impl PyGrandLayer {
    fn layer(&self) -> &CoverLayerReader<DefaultLabeledCloud<L2>> {
        self.tree.layer(self.scale_index)
    }
}

#[pymethods]
impl PyGrandLayer {
    pub fn radius(&self) -> f32 {
        self.parameters.scale_base.powi(self.layer().scale_index())
    }
    pub fn scale_index(&self) -> i32 {
        self.scale_index
    }
    pub fn len(&self) -> usize {
        self.layer().node_count()
    }
    pub fn center_indexes(&self) -> Vec<usize> {
        self.layer().map_nodes(|pi, _n| *pi)
    }
    pub fn child_addresses(&self, point_index: usize) -> Option<Vec<(i32, usize)>> {
        self.layer()
            .get_node_children_and(point_index, |nested_address, child_addresses| {
                let mut v = vec![nested_address];
                v.extend(child_addresses);
                v
            })
    }
    pub fn singleton_indexes(&self, point_index: usize) -> Option<Vec<usize>> {
        self.layer()
            .get_node_and(point_index, |n| Vec::from(n.singletons()))
    }

    pub fn is_leaf(&self, point_index: usize) -> Option<bool> {
        self.layer().get_node_and(point_index, |n| n.is_leaf())
    }

    pub fn centers(&self) -> PyResult<(Py<PyArray1<usize>>, Py<PyArray2<f32>>)> {
        let dim = self.parameters.point_cloud.dim();
        let mut centers =
            Vec::with_capacity(self.layer().node_count() * self.parameters.point_cloud.dim());
        let mut centers_indexes = Vec::with_capacity(self.layer().node_count());
        self.layer().for_each_node(|pi, _n| {
            centers_indexes.push(*pi);
            centers.extend(self.parameters.point_cloud.point(*pi).unwrap().dense_iter(dim));
        });
        let py_center_indexes = Array::from(centers_indexes);
        let py_centers = Array2::from_shape_vec(
            (self.layer().node_count(), self.parameters.point_cloud.dim()),
            centers,
        )
        .unwrap();
        let gil = GILGuard::acquire();
        let py = gil.python();
        Ok((
            py_center_indexes.into_pyarray(py).to_owned(),
            py_centers.into_pyarray(py).to_owned(),
        ))
    }

    pub fn child_points(&self, point_index: usize) -> PyResult<Option<Py<PyArray2<f32>>>> {
        let dim = self.parameters.point_cloud.dim();
        Ok(self
            .layer()
            .get_node_children_and(point_index, |nested_address, child_addresses| {
                let count = child_addresses.len() + 1;
                let mut centers: Vec<f32> = Vec::with_capacity(count * dim);
                centers.extend(
                    self.parameters
                        .point_cloud
                        .point(nested_address.1)
                        .unwrap()
                        .dense_iter(dim),
                );
                for na in child_addresses {
                    centers.extend(self.parameters.point_cloud.point(na.1).unwrap().dense_iter(dim));
                }
                let py_centers = Array2::from_shape_vec((count, dim), centers).unwrap();
                let gil = GILGuard::acquire();
                let py = gil.python();
                py_centers.into_pyarray(py).to_owned()
            }))
    }
    pub fn singleton_points(&self, point_index: usize) -> PyResult<Option<Py<PyArray2<f32>>>> {
        let dim = self.parameters.point_cloud.dim();
        Ok(self.layer().get_node_and(point_index, |node| {
            let singletons = node.singletons();
            let mut centers: Vec<f32> = Vec::with_capacity(singletons.len() * dim);
            for pi in singletons {
                centers.extend(self.parameters.point_cloud.point(*pi).unwrap().dense_iter(dim));
            }
            let py_centers = Array2::from_shape_vec((singletons.len(), dim), centers).unwrap();
            let gil = GILGuard::acquire();
            let py = gil.python();
            py_centers.into_pyarray(py).to_owned()
        }))
    }

    pub fn node(&self, center_index: usize) -> PyResult<PyGrandNode> {
        Ok(PyGrandNode {
            parameters: Arc::clone(&self.parameters),
            address: (self.scale_index, center_index),
            tree: Arc::clone(&self.tree),
        })
    }

    pub fn nodes(&self) -> PyResult<IterLayerNode> {
        Ok(IterLayerNode {
            parameters: Arc::clone(&self.parameters),
            addresses: self
                .layer()
                .node_center_indexes()
                .iter()
                .map(|pi| (self.scale_index, *pi))
                .collect(),
            tree: Arc::clone(&self.tree),
            index: 0,
        })
    }
}
