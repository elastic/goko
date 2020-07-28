use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::PyObjectProtocol;

use goko::plugins::distributions::*;
use goko::*;
use pointcloud::*;

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

#[pyclass(unsendable)]
pub struct PyBayesCategoricalTracker {
    pub hkl: BayesCategoricalTracker<DefaultLabeledCloud<L2>>,
    pub tree: CoverTreeReader<DefaultLabeledCloud<L2>>,
}

#[pymethods]
impl PyBayesCategoricalTracker {
    pub fn push(&mut self, point: &PyArray1<f32>) {
        let results = self
            .tree
            .path(point.readonly().as_slice().unwrap())
            .unwrap();
        self.hkl.add_path(results);
    }

    pub fn print(&self) {
        println!("{:#?}", self.hkl);
    }

    pub fn all_kl(&self) -> Vec<(f64, (i32, usize))> {
        self.hkl.all_node_kl()
    }
    pub fn stats(&self) -> PyKLDivergenceStats {
        PyKLDivergenceStats {
            stats: self.hkl.current_stats(),
        }
    }
}

#[pyclass(unsendable)]
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
        self.stats.nz_count
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
        self.stats.sequence_len
    }
    #[getter]
    pub fn layer_totals(&self) -> Vec<u64> {
        self.stats.layer_totals.clone()
    }
    #[getter]
    pub fn weighed_layer_totals(&self) -> Vec<f32> {
        self.stats.weighted_layer_totals.clone()
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
