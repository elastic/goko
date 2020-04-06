use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::PyObjectProtocol;

use grandma::plugins::distributions::*;
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
pub struct PyBayesCategoricalTracker {
    pub hkl: BayesCategoricalTracker<L2>,
    pub tree: Arc<CoverTreeReader<L2>>,
}

#[pymethods]
impl PyBayesCategoricalTracker {
    pub fn push(&mut self, point: &PyArray1<f32>) {
        let results = self.tree.dry_insert(point.as_slice().unwrap()).unwrap();
        self.hkl
            .add_dry_insert(results);
    }

    pub fn print(&self) {
        println!("{:#?}", self.hkl);
    }

    pub fn all_kl(&self) -> Vec<(f64, (i32, u64))> {
        self.hkl.all_node_kl()
    }
    pub fn stats(&self) -> PyKLDivergenceStats {
        PyKLDivergenceStats {
            stats: self.hkl.current_stats(),
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
    pub fn max(&self) -> f64 {
        self.stats.max
    }
    #[getter]
    pub fn min(&self) -> f64 {
        self.stats.min
    }
    #[getter]
    pub fn nz_count(&self) -> u64 {
        self.stats.nz_count as u64
    }
    #[getter]
    pub fn moment1_nz(&self) -> f64 {
        self.stats.moment1_nz
    }
    #[getter]
    pub fn moment2_nz(&self) -> f64 {
        self.stats.moment2_nz
    }
    #[getter]
    pub fn sequence_len(&self) -> u64 {
        self.stats.sequence_len as u64
    }
}

#[pyproto]
impl PyObjectProtocol for PyKLDivergenceStats {
    fn __repr__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.stats))
    }
    fn __str__(&self) -> PyResult<String> {
        Ok(format!("{:?}", self.stats))
    }
}
