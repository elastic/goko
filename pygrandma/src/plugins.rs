use pyo3::prelude::*;
use pyo3::PyObjectProtocol;
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
    pub fn stats(&self) -> PyKLDivergenceStats {
        PyKLDivergenceStats {
            stats:self.hkl.current_stats()
        }
    }
}

#[pyclass]
pub struct PyKLDivergenceStats {
    pub stats: KLDivergenceStats,
}

#[pymethods]
impl PyKLDivergenceStats {
    #[getter]
    pub fn mean_max(&self) -> f64 {
        self.stats.mean_max()
    }
    #[getter]
    pub fn var_max(&self) -> f64 {
        self.stats.var_max()
    }
    #[getter]
    pub fn mean_min(&self) -> f64 {
        self.stats.mean_min()
    }
    #[getter]
    pub fn var_min(&self) -> f64 {
        self.stats.var_min()
    }
    #[getter]
    pub fn mean_nz_count(&self) -> f64 {
        self.stats.mean_nz_count()
    }
    #[getter]
    pub fn var_nz_count(&self) -> f64 {
        self.stats.var_nz_count()
    }
    #[getter]
    pub fn mean_mean(&self) -> f64 {
        self.stats.mean_mean()
    }
    #[getter]
    pub fn var_mean(&self) -> f64 {
        self.stats.var_mean()
    }
    #[getter]
    pub fn mean_nz(&self) -> f64 {
        self.stats.mean_nz()
    }
    #[getter]
    pub fn var_nz(&self) -> f64 {
        self.stats.var_nz()
    }
    #[getter]
    pub fn nz_total_count(&self) -> u64 {
        self.stats.nz_total_count as u64
    }
    #[getter]
    pub fn sequence_count(&self) -> u64 {
        self.stats.sequence_count as u64
    }
    #[getter]
    pub fn sequence_len(&self) -> u64 {
        self.stats.sequence_len as u64
    }
}

#[pyproto]
impl PyObjectProtocol for PyKLDivergenceStats{
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}",self.stats))
    }
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}",self.stats))
    }
}