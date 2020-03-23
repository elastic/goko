use pyo3::prelude::*;

use ndarray::{Array, Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::{PyIterProtocol};

use grandma::layer::*;
use grandma::plugins::*;
use grandma::*;
use grandma::errors::GrandmaError;
use pointcloud::*;
use std::sync::Arc;

use rayon::prelude::*;
use crate::tree::*;
use crate::layer::*;

#[pyclass(module = "py_egs_events")]
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

#[pyclass(module = "pygrandma")]
pub struct PyGrandNode {
    pub parameters: Arc<CoverTreeParameters<L2>>,
    pub address: NodeAddress,
    pub tree: Arc<CoverTreeReader<L2>>,
}

#[pymethods]
impl PyGrandNode {
    pub fn address(&self) -> (i32,u64) {
        self.address
    }
    pub fn cover_mean(&self) -> PyResult<Option<Py<PyArray1<f32>>>> {
        let dim = self.parameters.point_cloud.dim();
        let gil = GILGuard::acquire();
        let py = gil.python();
        let mean = self
            .tree
            .get_node_plugin_and::<DiagGaussianNode, _, _>(self.address, |p| p.mean())
            .map(|m| Array1::from_shape_vec((dim,), m).unwrap().into_pyarray(py).to_owned());

        
        Ok(mean)
    }

    pub fn cover_diag_var(&self) -> PyResult<Py<PyArray1<f32>>> {
        let dim = self.parameters.point_cloud.dim();
        let var = self
            .tree
            .get_node_plugin_and::<DiagGaussianNode, _, _>(self.address, |p| p.var())
            .unwrap();
        let py_mean = Array1::from_shape_vec((dim,), var).unwrap();
        let gil = GILGuard::acquire();
        let py = gil.python();
        Ok(py_mean.into_pyarray(py).to_owned())
    }
}