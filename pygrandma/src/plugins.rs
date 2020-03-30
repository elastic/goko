use pyo3::prelude::*;

use numpy::PyArray1;

use grandma::plugins::utils::*;
use grandma::*;
use pointcloud::*;
use std::sync::Arc;

/*
pub #[derive(Debug)]
struct PyBucketProbs {
    probs: BucketProbs
}

#[pymethods]
impl PyBucketProbs {
    pub fn pdfs(&self) -> PyResult<Py<PyArray1<f32>>> {
        Array1::from_shape_vec((dim,), m)
            .unwrap()
            .into_pyarray(py)
            .to_owned()
    }
}
*/

#[pyclass]
pub struct PyBucketHKLDivergence {
    pub hkl: BucketHKLDivergence<L2>,
    pub tree: Arc<CoverTreeReader<L2>>,
}

#[pymethods]
impl PyBucketHKLDivergence {
    pub fn push(&mut self, point: &PyArray1<f32>) {
        let results = self.tree.dry_insert(point.as_slice().unwrap()).unwrap();
        self.hkl
            .add_trace(results.iter().map(|(_, a)| *a).collect());
    }

    pub fn all_kl(&self) -> Vec<(f64, (i32, u64))> {
        self.hkl.all_node_kl()
    }
}

#[pyclass]
pub struct PySGDHKLDivergence {
    pub hkl: SGDHKLDivergence<L2>,
    pub tree: Arc<CoverTreeReader<L2>>,
}

#[pymethods]
impl PySGDHKLDivergence {
    pub fn push(&mut self, point: &PyArray1<f32>) {
        let results = self.tree.dry_insert(point.as_slice().unwrap()).unwrap();
        self.hkl
            .add_trace(results.iter().map(|(_, a)| *a).collect());
    }

    pub fn print(&self) {
        println!("{:#?}", self.hkl);
    }

    pub fn all_kl(&self) -> Vec<(f64, (i32, u64))> {
        self.hkl.all_node_kl()
    }
}
