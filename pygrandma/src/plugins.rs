use pyo3::prelude::*;

use ndarray::{Array, Array1, Array2};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::PyIterProtocol;

use grandma::plugins::utils::*;
use grandma::*;
use pointcloud::*;
use std::sync::Arc;
#[pyclass(module = "py_egs_events")]
pub struct PyBucketHKLDivergence {
    hkl: BucketHKLDivergence,
    tree: Arc<CoverTreeReader<L2>>,
}

#[pymethods]
impl PyBucketHKLDivergence {
    pub fn insert(&mut self, point: &PyArray1<f32>) {
        let results = self.tree.dry_insert(point.as_slice().unwrap()).unwrap();
        self.hkl
            .add_trace(results.iter().map(|(_, a)| *a).collect());
    }
}
