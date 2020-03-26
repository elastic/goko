use pyo3::prelude::*;

use numpy::PyArray1;

use grandma::plugins::utils::*;
use grandma::*;
use pointcloud::*;
use std::sync::Arc;
#[pyclass(module = "py_egs_events")]
pub struct PyBucketHKLDivergence {
    pub hkl: BucketHKLDivergence,
    pub tree: Arc<CoverTreeReader<L2>>,
}

#[pymethods]
impl PyBucketHKLDivergence {
    pub fn insert(&mut self, point: &PyArray1<f32>) {
        let results = self.tree.dry_insert(point.as_slice().unwrap()).unwrap();
        self.hkl
            .add_trace(results.iter().map(|(_, a)| *a).collect());
    }

    pub fn all_kl(&self) -> Vec<(f32,(i32,u64))> {
        self.hkl.all_node_kl(&self.tree)
    }
}
