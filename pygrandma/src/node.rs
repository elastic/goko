use pyo3::prelude::*;

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::PyIterProtocol;

use grandma::plugins::distributions::*;
use grandma::*;
use pointcloud::*;
use std::sync::Arc;

#[pyclass]
pub struct IterLayerNode {
    pub parameters: Arc<CoverTreeParameters<L2>>,
    pub addresses: Vec<NodeAddress>,
    pub tree: Arc<CoverTreeReader<L2>>,
    pub index: usize,
}

impl std::iter::Iterator for IterLayerNode {
    type Item = PyGrandNode;
    fn next(&mut self) -> Option<PyGrandNode> {
        if self.index < self.addresses.len() {
            let index = self.index;
            self.index += 1;
            Some(PyGrandNode {
                parameters: Arc::clone(&self.parameters),
                address: self.addresses[index],
                tree: Arc::clone(&self.tree),
            })
        } else {
            None
        }
    }
}

#[pyproto]
impl PyIterProtocol for IterLayerNode {
    fn __iter__(slf: PyRefMut<Self>) -> PyResult<Py<IterLayerNode>> {
        Ok(slf.into())
    }
    fn __next__(mut slf: PyRefMut<Self>) -> PyResult<Option<PyGrandNode>> {
        Ok(slf.next())
    }
}

#[pyclass]
pub struct PyGrandNode {
    pub parameters: Arc<CoverTreeParameters<L2>>,
    pub address: NodeAddress,
    pub tree: Arc<CoverTreeReader<L2>>,
}

#[pymethods]
impl PyGrandNode {
    pub fn address(&self) -> (i32, u64) {
        self.address
    }

    pub fn coverage_count(&self) -> u64 {
        self.tree
            .get_node_plugin_and::<Dirichlet, _, _>(self.address, |p| p.total())
            .unwrap() as u64
    }

    pub fn children(&self) -> Vec<PyGrandNode> {
        self.children_addresses()
            .iter()
            .map(|address| PyGrandNode {
                parameters: Arc::clone(&self.parameters),
                address: *address,
                tree: Arc::clone(&self.tree),
            })
            .collect()
    }

    pub fn children_addresses(&self) -> Vec<(i32, u64)> {
        self.tree
            .get_node_and(self.address, |n| {
                n.children().map(|(nested_scale, children)| {
                    let mut py_nodes: Vec<(i32, u64)> = Vec::from(children);
                    py_nodes.push((nested_scale, *n.center_index()));
                    py_nodes
                })
            })
            .flatten()
            .unwrap_or(vec![])
    }

    pub fn fractal_dim(&self) -> f32 {
        self.tree.node_fractal_dim(self.address)
    }

    pub fn weighted_fractal_dim(&self) -> f32 {
        self.tree.node_weighted_fractal_dim(self.address)
    }

    pub fn singletons(&self) -> PyResult<Py<PyArray2<f32>>> {
        let dim = self.parameters.point_cloud.dim();
        let len = self.coverage_count() as usize;
        let mut ret_matrix = Vec::with_capacity(len * dim);
        self.tree.get_node_and(self.address, |n| {
            n.singletons().iter().for_each(|pi| {
                ret_matrix.extend(self.parameters.point_cloud.get_point(*pi).unwrap_or(&[]));
            });

            if n.is_leaf() {
                ret_matrix.extend(
                    self.parameters
                        .point_cloud
                        .get_point(*n.center_index())
                        .unwrap_or(&[]),
                );
            }
        });

        let ret_matrix = Array2::from_shape_vec((len, dim), ret_matrix).unwrap();
        let gil = GILGuard::acquire();
        let py = gil.python();
        Ok(ret_matrix.into_pyarray(py).to_owned())
    }

    pub fn singletons_indexes(&self) -> Vec<u64> {
        self.tree
            .get_node_and(self.address, |n| Vec::from(n.singletons()))
            .unwrap_or(vec![])
    }

    pub fn cover_mean(&self) -> PyResult<Option<Py<PyArray1<f32>>>> {
        let dim = self.parameters.point_cloud.dim();
        let gil = GILGuard::acquire();
        let py = gil.python();
        let mean = self
            .tree
            .get_node_plugin_and::<DiagGaussian, _, _>(self.address, |p| p.mean())
            .map(|m| {
                Array1::from_shape_vec((dim,), m)
                    .unwrap()
                    .into_pyarray(py)
                    .to_owned()
            });

        Ok(mean)
    }

    pub fn cover_diag_var(&self) -> PyResult<Py<PyArray1<f32>>> {
        let dim = self.parameters.point_cloud.dim();
        let var = self
            .tree
            .get_node_plugin_and::<DiagGaussian, _, _>(self.address, |p| p.var())
            .unwrap();
        let py_mean = Array1::from_shape_vec((dim,), var).unwrap();
        let gil = GILGuard::acquire();
        let py = gil.python();
        Ok(py_mean.into_pyarray(py).to_owned())
    }
}
