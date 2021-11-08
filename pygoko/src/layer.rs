use pyo3::prelude::*;

use ndarray::{Array, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::PyIterProtocol;

use goko::layer::*;
use goko::*;
use pointcloud::*;
use std::sync::Arc;

use crate::node::*;

#[pyclass(unsendable)]
pub struct IterLayers {
    pub parameters: Arc<CoverTreeParameters<DefaultLabeledCloud<L2>>>,
    pub tree: CoverTreeReader<DefaultLabeledCloud<L2>>,
    pub scale_indexes: Vec<i32>,
    pub index: usize,
}

impl std::iter::Iterator for IterLayers {
    type Item = PyLayer;
    fn next(&mut self) -> Option<PyLayer> {
        if self.index < self.scale_indexes.len() {
            self.index += 1;
            Some(PyLayer {
                parameters: Arc::clone(&self.parameters),
                tree: self.tree.clone(),
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
    fn __next__(mut slf: PyRefMut<Self>) -> PyResult<Option<PyLayer>> {
        Ok(slf.next())
    }
}

#[pyclass(unsendable)]
pub struct PyLayer {
    pub parameters: Arc<CoverTreeParameters<DefaultLabeledCloud<L2>>>,
    pub tree: CoverTreeReader<DefaultLabeledCloud<L2>>,
    pub scale_index: i32,
}

impl PyLayer {
    fn layer(&self) -> &CoverLayerReader<DefaultLabeledCloud<L2>> {
        self.tree.layer(self.scale_index)
    }
}

#[pymethods]
impl PyLayer {
    pub fn radius(&self) -> f32 {
        self.parameters.scale_base.powi(self.layer().scale_index())
    }
    pub fn scale_index(&self) -> i32 {
        self.scale_index
    }
    pub fn len(&self) -> usize {
        self.layer().len()
    }
    pub fn center_indexes(&self) -> Vec<usize> {
        self.layer().map_nodes(|pi, _n| *pi)
    }
    pub fn child_addresses(&self, point_index: usize) -> Option<Vec<(i32, usize)>> {
        self.layer()
            .get_node_children_and(point_index, |child_addresses| {
                child_addresses
                    .to_tuples()
                    .drain(0..)
                    .filter_map(|p| p)
                    .collect()
            })
    }
    pub fn singleton_indexes(&self, point_index: usize) -> Option<Vec<usize>> {
        self.layer()
            .get_node_and(point_index, |n| Vec::from(n.singletons()))
    }

    pub fn is_leaf(&self, point_index: usize) -> Option<bool> {
        self.layer().get_node_and(point_index, |n| n.is_leaf())
    }

    pub fn fractal_dim(&self) -> f32 {
        self.tree.layer_fractal_dim(self.scale_index)
    }

    pub fn weighted_fractal_dim(&self) -> f32 {
        self.tree.layer_weighted_fractal_dim(self.scale_index)
    }

    pub fn centers(&self) -> PyResult<(Py<PyArray1<usize>>, Py<PyArray2<f32>>)> {
        let mut centers =
            Vec::with_capacity(self.layer().len() * self.parameters.point_cloud.dim());
        let mut centers_indexes = Vec::with_capacity(self.layer().len());
        self.layer().for_each_node(|pi, _n| {
            centers_indexes.push(*pi);
            centers.extend(self.parameters.point_cloud.point(*pi).unwrap().dense_iter());
        });
        let py_center_indexes = Array::from(centers_indexes);
        let py_centers = Array2::from_shape_vec(
            (self.layer().len(), self.parameters.point_cloud.dim()),
            centers,
        )
        .unwrap();
        let gil = pyo3::Python::acquire_gil();
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
            .get_node_children_and(point_index, |child_addresses| {
                let count = child_addresses.len();
                let mut centers: Vec<f32> = Vec::with_capacity(count * dim);
                for na in child_addresses {
                    centers.extend(
                        self.parameters
                            .point_cloud
                            .point(na.point_index())
                            .unwrap()
                            .dense_iter(),
                    );
                }
                let py_centers = Array2::from_shape_vec((count, dim), centers).unwrap();
                let gil = pyo3::Python::acquire_gil();
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
                centers.extend(self.parameters.point_cloud.point(*pi).unwrap().dense_iter());
            }
            let py_centers = Array2::from_shape_vec((singletons.len(), dim), centers).unwrap();
            let gil = pyo3::Python::acquire_gil();
            let py = gil.python();
            py_centers.into_pyarray(py).to_owned()
        }))
    }

    pub fn node(&self, center_index: usize) -> PyResult<PyNode> {
        Ok(PyNode {
            parameters: Arc::clone(&self.parameters),
            address: (self.scale_index, center_index).into(),
            tree: self.tree.clone(),
        })
    }

    pub fn nodes(&self) -> PyResult<IterLayerNode> {
        Ok(IterLayerNode {
            parameters: Arc::clone(&self.parameters),
            addresses: self
                .layer()
                .node_center_indexes()
                .iter()
                .map(|pi| (self.scale_index, *pi).into())
                .collect(),
            tree: self.tree.clone(),
            index: 0,
        })
    }
}
