use goko::plugins::discrete::prelude::*;
use goko::*;
use numpy::PyArray1;
use pointcloud::*;
use pyo3::prelude::*;
use pyo3::types::PyDict;

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
            .path(&point.readonly().as_slice().unwrap())
            .unwrap();
        self.hkl.add_path(results);
    }

    pub fn print(&self) {
        println!("{:#?}", self.hkl);
    }

    pub fn probs(&self, node_address: (i32, usize)) -> Option<(Vec<((i32, usize), f64)>, f64)> {
        self.hkl.prob_vector(node_address)
    }

    pub fn evidence(&self, node_address: (i32, usize)) -> Option<(Vec<((i32, usize), f64)>, f64)> {
        self.hkl.evidence_prob_vector(node_address)
    }

    pub fn all_kl(&self) -> Vec<(f64, (i32, usize))> {
        self.hkl.all_node_kl()
    }
    pub fn stats(&self) -> PyResult<PyObject> {
        let stats = self.hkl.kl_div_stats();
        let gil = pyo3::Python::acquire_gil();
        let py = gil.python();
        let dict = PyDict::new(py);
        dict.set_item("max", stats.max)?;
        dict.set_item("min", stats.min)?;
        dict.set_item("nz_count", stats.nz_count)?;
        dict.set_item("moment1_nz", stats.moment1_nz)?;
        dict.set_item("moment2_nz", stats.moment2_nz)?;
        dict.set_item("sequence_len", stats.sequence_len)?;
        Ok(dict.into())
    }
}

#[pyclass(unsendable)]
pub struct PyKLDivergenceBaseline {
    pub baseline: KLDivergenceBaseline,
}

#[pymethods]
impl PyKLDivergenceBaseline {
    pub fn stats(&self, i: usize) -> PyResult<PyObject> {
        let stats = self.baseline.stats(i);
        let gil = pyo3::Python::acquire_gil();
        let dict = PyDict::new(gil.python());
        let max_dict = PyDict::new(gil.python());
        max_dict.set_item("mean", stats.max.0)?;
        max_dict.set_item("var", stats.max.1)?;
        dict.set_item("max", max_dict)?;

        let min_dict = PyDict::new(gil.python());
        min_dict.set_item("mean", stats.min.0)?;
        min_dict.set_item("var", stats.min.1)?;
        dict.set_item("min", min_dict)?;

        let nz_count_dict = PyDict::new(gil.python());
        nz_count_dict.set_item("mean", stats.nz_count.0)?;
        nz_count_dict.set_item("var", stats.nz_count.1)?;
        dict.set_item("nz_count", nz_count_dict)?;

        let moment1_nz_dict = PyDict::new(gil.python());
        moment1_nz_dict.set_item("mean", stats.moment1_nz.0)?;
        moment1_nz_dict.set_item("var", stats.moment1_nz.1)?;
        dict.set_item("moment1_nz", moment1_nz_dict)?;

        let moment2_nz_dict = PyDict::new(gil.python());
        moment2_nz_dict.set_item("mean", stats.moment2_nz.0)?;
        moment2_nz_dict.set_item("var", stats.moment2_nz.1)?;
        dict.set_item("moment2_nz", moment2_nz_dict)?;
        Ok(dict.into())
    }
}
