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
    /*
    pub fn probs(&self, node_address: Option<(i32, usize)>) -> Option<Vec<(Option<(i32, usize)>, f64)>> {
        if let Some(na) = node_address {
            self.hkl.node_tracker(na.into()).map(|diri| diri.prior().param_vec()).map(|pvec| pvec.to_tuples())
        } else {
            Some(self.hkl.overall_tracker().prior().param_vec().to_tuples())
        }
    }

    pub fn evidence(&self, node_address: Option<(i32, usize)>) -> Option<Vec<(Option<(i32, usize)>, f64)>> {
        if let Some(na) = node_address {
            self.hkl.node_tracker(na.into()).map(|diri| diri.data().data_vec()).map(|pvec| pvec.to_tuples())
        } else {
            Some(self.hkl.overall_tracker().data().data_vec().to_tuples())
        }
    }
    */
    pub fn nodes_kl_div(&self) -> Vec<(Option<(i32, usize)>, f64)> {
        self.hkl.nodes_kl_div().to_tuples()
    }

    pub fn overall_kl_div(&self) -> f64 {
        self.hkl.overall_kl_div()
    }

    pub fn nodes_kl_div_stats(&self) -> PyResult<PyObject> {
        let stats = self.hkl.nodes_kl_div_stats();
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

    pub fn nodes_aic(&self) -> Vec<(Option<(i32, usize)>, f64)> {
        self.hkl.nodes_aic().to_tuples()
    }

    pub fn overall_aic(&self) -> f64 {
        self.hkl.marginal_aic()
    }

    pub fn nodes_aic_stats(&self) -> PyResult<PyObject> {
        let stats = self.hkl.nodes_aic_stats();
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

    pub fn nodes_mll(&self) -> Vec<(Option<(i32, usize)>, f64)> {
        self.hkl.nodes_mll().to_tuples()
    }

    pub fn overall_mll(&self) -> f64 {
        self.hkl.overall_mll()
    }

    pub fn nodes_mll_stats(&self) -> PyResult<PyObject> {
        let stats = self.hkl.nodes_mll_stats();
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

    pub fn nodes_corrected_mll(&self) -> Vec<(Option<(i32, usize)>, f64)> {
        self.hkl.nodes_corrected_mll().to_tuples()
    }

    pub fn nodes_corrected_mll_stats(&self) -> PyResult<PyObject> {
        let stats = self.hkl.nodes_corrected_mll_stats();
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


    pub fn overall_corrected_mll(&self) -> f64 {
        self.hkl.overall_corrected_mll()
    }
}
