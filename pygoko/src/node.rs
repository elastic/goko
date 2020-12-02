use pyo3::prelude::*;

use ndarray::{Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::PyIterProtocol;

use goko::plugins::discrete::prelude::*;
use goko::plugins::gaussians::*;
use goko::*;
use pointcloud::*;
use std::sync::Arc;

use pyo3::types::PyDict;

#[pyclass(unsendable)]
pub struct IterLayerNode {
    pub parameters: Arc<CoverTreeParameters<DefaultLabeledCloud<L2>>>,
    pub addresses: Vec<NodeAddress>,
    pub tree: CoverTreeReader<DefaultLabeledCloud<L2>>,
    pub index: usize,
}

impl std::iter::Iterator for IterLayerNode {
    type Item = PyNode;
    fn next(&mut self) -> Option<PyNode> {
        if self.index < self.addresses.len() {
            let index = self.index;
            self.index += 1;
            Some(PyNode {
                parameters: Arc::clone(&self.parameters),
                address: self.addresses[index],
                tree: self.tree.clone(),
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
    fn __next__(mut slf: PyRefMut<Self>) -> PyResult<Option<PyNode>> {
        Ok(slf.next())
    }
}

#[pyclass(unsendable)]
pub struct PyNode {
    pub parameters: Arc<CoverTreeParameters<DefaultLabeledCloud<L2>>>,
    pub address: NodeAddress,
    pub tree: CoverTreeReader<DefaultLabeledCloud<L2>>,
}

#[pymethods]
impl PyNode {
    pub fn address(&self) -> (i32, usize) {
        self.address
    }

    pub fn is_leaf(&self) -> bool {
        self.tree
            .get_node_and(self.address, |n| n.is_leaf())
            .unwrap()
    }

    pub fn coverage_count(&self) -> usize {
        self.tree
            .get_node_and(self.address, |n| n.coverage_count())
            .unwrap()
    }

    pub fn children(&self) -> Vec<PyNode> {
        self.children_addresses()
            .iter()
            .map(|address| PyNode {
                parameters: Arc::clone(&self.parameters),
                address: *address,
                tree: self.tree.clone(),
            })
            .collect()
    }

    pub fn children_probs(&self) -> Option<(Vec<((i32, usize), f64)>, f64)> {
        self.tree
            .get_node_plugin_and(self.address, |p: &Dirichlet| p.prob_vector())
            .flatten()
    }

    pub fn children_addresses(&self) -> Vec<(i32, usize)> {
        self.tree
            .get_node_and(self.address, |n| {
                n.children().map(|(nested_scale, children)| {
                    let mut py_nodes: Vec<(i32, usize)> = Vec::from(children);
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
                if let Ok(p) = self.parameters.point_cloud.point(*pi) {
                    ret_matrix.extend(p.dense_iter());
                }
            });

            if n.is_leaf() {
                if let Ok(p) = self.parameters.point_cloud.point(*n.center_index()) {
                    ret_matrix.extend(p.dense_iter());
                }
            }
        });

        let ret_matrix = Array2::from_shape_vec((len, dim), ret_matrix).unwrap();
        let gil = GILGuard::acquire();
        let py = gil.python();
        Ok(ret_matrix.into_pyarray(py).to_owned())
    }

    pub fn singletons_indexes(&self) -> Vec<usize> {
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

    pub fn get_singular_values(&self) -> PyResult<Option<Py<PyArray1<f32>>>> {
        let gil = GILGuard::acquire();
        let py = gil.python();
        Ok(self
            .tree
            .get_node_plugin_and::<SvdGaussian, _, _>(self.address, |p| {
                p.singular_vals.clone().into_pyarray(py).to_owned()
            }))
    }

    pub fn label_summary(&self) -> PyResult<Option<PyObject>> {
        let gil = GILGuard::acquire();
        let py = gil.python();
        let dict = PyDict::new(py);
        match self.tree.get_node_label_summary(self.address) {
            Some(s) => {
                dict.set_item("errors", s.errors)?;
                dict.set_item("nones", s.nones)?;
                dict.set_item("items", s.summary.items.to_vec())?;
                Ok(Some(dict.into()))
            }
            None => Ok(None),
        }
    }
}
