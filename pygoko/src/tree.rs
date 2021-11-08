/*
* Licensed to Elasticsearch B.V. under one or more contributor
* license agreements. See the NOTICE file distributed with
* this work for additional information regarding copyright
* ownership. Elasticsearch B.V. licenses this file to you under
* the Apache License, Version 2.0 (the "License"); you may
* not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*  http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing,
* software distributed under the License is distributed on an
* "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
* KIND, either express or implied.  See the License for the
* specific language governing permissions and limitations
* under the License.
*/

use ndarray::Array1;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use rand::prelude::*;

use std::path::Path;
use std::sync::Arc;

use goko::query_interface::BulkInterface;
use goko::*;
use pointcloud::loaders::labeled_ram_from_yaml;
use pointcloud::*;

use crate::layer::*;
use crate::node::*;
use crate::plugins::*;
use goko::plugins::discrete::prelude::*;
use goko::plugins::gaussians::*;

#[pyclass(unsendable)]
pub struct CoverTree {
    builder: Option<CoverTreeBuilder>,
    temp_point_cloud: Option<Arc<DefaultLabeledCloud<L2>>>,
    writer: Option<CoverTreeWriter<DefaultLabeledCloud<L2>>>,
    metric: String,
}

#[pymethods]
impl CoverTree {
    #[new]
    fn new() -> PyResult<CoverTree> {
        Ok(CoverTree {
            builder: Some(CoverTreeBuilder::new()),
            temp_point_cloud: None,
            writer: None,
            metric: "DefaultLabeledCloud<L2>".to_string(),
        })
    }
    pub fn set_scale_base(&mut self, x: f32) {
        match &mut self.builder {
            Some(builder) => builder.set_scale_base(x),
            None => panic!("Set too late"),
        };
    }
    pub fn set_leaf_cutoff(&mut self, x: usize) {
        match &mut self.builder {
            Some(builder) => builder.set_leaf_cutoff(x),
            None => panic!("Set too late"),
        };
    }
    pub fn set_min_res_index(&mut self, x: i32) {
        match &mut self.builder {
            Some(builder) => builder.set_min_res_index(x),
            None => panic!("Set too late"),
        };
    }
    pub fn set_use_singletons(&mut self, x: bool) {
        match &mut self.builder {
            Some(builder) => builder.set_use_singletons(x),
            None => panic!("Set too late"),
        };
    }

    pub fn set_verbosity(&mut self, x: u32) {
        match &mut self.builder {
            Some(builder) => builder.set_verbosity(x),
            None => panic!("Set too late"),
        };
    }

    pub fn load_yaml_config(&mut self, file_name: String) -> PyResult<()> {
        let path = Path::new(&file_name);
        let point_cloud = Arc::new(labeled_ram_from_yaml::<_, L2>(&path).unwrap());
        let builder = CoverTreeBuilder::from_yaml(&path);
        self.builder = Some(builder);
        self.temp_point_cloud = Some(point_cloud);
        Ok(())
    }

    pub fn set_metric(&mut self, metric_name: String) {
        self.metric = metric_name;
    }

    pub fn fit(
        &mut self,
        data: Option<&PyArray2<f32>>,
        labels: Option<&PyArray1<i64>>,
    ) -> PyResult<()> {
        let point_cloud = if let Some(data) = data {
            let len = data.shape()[0];
            let data_dim = data.shape()[1];
            let my_labels: Vec<i64> = match labels {
                Some(labels) => Vec::from(labels.readonly().as_slice().unwrap()),
                None => vec![0; len],
            };
            Arc::new(DefaultLabeledCloud::<L2>::new_simple(
                Vec::from(data.readonly().as_slice().unwrap()),
                data_dim,
                my_labels,
            ))
        } else {
            if let Some(point_cloud) = self.temp_point_cloud.take() {
                point_cloud
            } else {
                panic!("No known point_cloud");
            }
        };

        let builder = self.builder.take();
        self.writer = Some(builder.unwrap().build(point_cloud).unwrap());
        let writer = self.writer.as_mut().unwrap();
        writer.generate_summaries();
        //writer.add_plugin::<GokoDiagGaussian>(GokoDiagGaussian::singletons());
        writer.add_plugin::<GokoDirichlet>(GokoDirichlet {});
        Ok(())
    }

    /*
    pub fn attach_svds(&mut self, min_point_count: usize, max_point_count: usize, tau: f32) {
        let writer = self.writer.as_mut().unwrap();
        writer.add_plugin::<GokoSvdGaussian>(GokoSvdGaussian::new(
            min_point_count,
            max_point_count,
            tau,
        ));
    }
    */

    pub fn data_point(&self, point_index: usize) -> PyResult<Option<Py<PyArray1<f32>>>> {
        let reader = self.writer.as_ref().unwrap().reader();
        let dim = reader.parameters().point_cloud.dim();
        Ok(match reader.parameters().point_cloud.point(point_index) {
            Err(_) => None,
            Ok(point) => {
                let py_point =
                    Array1::from_shape_vec((dim,), point.dense_iter().collect()).unwrap();
                let gil = pyo3::Python::acquire_gil();
                let py = gil.python();
                Some(py_point.into_pyarray(py).to_owned())
            }
        })
    }

    //pub fn layers(&self) ->
    pub fn top_scale(&self) -> Option<i32> {
        self.writer
            .as_ref()
            .map(|w| w.reader().scale_range().end - 1)
    }

    pub fn bottom_scale(&self) -> Option<i32> {
        self.writer.as_ref().map(|w| w.reader().scale_range().start)
    }

    pub fn scale_base(&self) -> Option<f32> {
        self.writer
            .as_ref()
            .map(|w| w.reader().parameters().scale_base)
    }

    pub fn layers(&self) -> PyResult<IterLayers> {
        let reader = self.writer.as_ref().unwrap().reader();
        let scale_indexes = reader.layers().map(|(si, _)| si).collect();
        Ok(IterLayers {
            parameters: Arc::clone(reader.parameters()),
            tree: reader,
            scale_indexes,
            index: 0,
        })
    }

    pub fn layer(&self, scale_index: i32) -> PyResult<PyLayer> {
        let reader = self.writer.as_ref().unwrap().reader();
        Ok(PyLayer {
            parameters: Arc::clone(reader.parameters()),
            tree: reader,
            scale_index,
        })
    }

    pub fn node(&self, address: (i32, usize)) -> PyResult<PyNode> {
        let reader = self.writer.as_ref().unwrap().reader();
        let na: NodeAddress = address.into();
        // Check node exists
        reader.get_node_and(na, |_| true).unwrap();
        Ok(PyNode {
            parameters: Arc::clone(reader.parameters()),
            address: na,
            tree: reader,
        })
    }

    pub fn root(&self) -> PyResult<PyNode> {
        let reader = self.writer.as_ref().unwrap().reader();
        self.node(reader.root_address().to_tuple().unwrap())
    }

    pub fn knn(&self, point: &PyArray1<f32>, k: usize) -> Vec<(usize, f32)> {
        let reader = self.writer.as_ref().unwrap().reader();
        reader
            .knn(&point.readonly().as_slice().unwrap(), k)
            .unwrap()
    }

    pub fn routing_knn(&self, point: &PyArray1<f32>, k: usize) -> Vec<(usize, f32)> {
        let reader = self.writer.as_ref().unwrap().reader();
        reader
            .routing_knn(&point.readonly().as_slice().unwrap(), k)
            .unwrap()
    }

    pub fn path(&self, point: &PyArray1<f32>) -> Vec<((i32, usize), f32)> {
        let reader = self.writer.as_ref().unwrap().reader();
        reader.path(&point.readonly().as_slice().unwrap()).unwrap().to_valid_tuples()
    }

    pub fn known_path(&self, point_index: usize) -> Vec<((i32, usize), f32)> {
        let reader = self.writer.as_ref().unwrap().reader();
        reader.known_path(point_index).unwrap().to_valid_tuples()
    }

    pub fn index_depths(&self, point_indexes: Vec<usize>, tau: Option<f32>) -> Vec<(usize, usize)> {
        let reader = self.writer.as_ref().unwrap().reader();
        let bulk = BulkInterface::new(reader);
        let tau = tau.unwrap_or(0.00001);
        bulk.known_path_and(&point_indexes, |reader, path| {
            if let Ok(path) = path {
                let mut homogenity_depth = path.len();
                for (i, (a, _d)) in path.iter().enumerate() {
                    let summ = reader.get_node_label_summary(*a).unwrap();
                    if summ.summary.items.len() == 1 {
                        homogenity_depth = i;
                        break;
                    }
                    let sum = summ.summary.items.iter().map(|(_, c)| c).sum::<usize>() as f32;
                    let max = *summ.summary.items.iter().map(|(_, c)| c).max().unwrap() as f32;
                    if 1.0 - max / sum < tau {
                        homogenity_depth = i;
                        break;
                    }
                }
                (path.len(), homogenity_depth)
            } else {
                (0, 0)
            }
        })
    }

    pub fn point_depths(&self, points: &PyArray2<f32>, tau: Option<f32>) -> Vec<(usize, usize)> {
        let reader = self.writer.as_ref().unwrap().reader();
        let bulk = BulkInterface::new(reader);
        let tau = tau.unwrap_or(0.00001);

        bulk.array_map_with_reader(points.readonly().as_array(), |reader, point| {
            if let Ok(path) = reader.path(point) {
                let mut homogenity_depth = path.len();
                for (i, (a, _d)) in path.iter().enumerate() {
                    let summ = reader.get_node_label_summary(*a).unwrap();
                    if summ.summary.items.len() == 1 {
                        homogenity_depth = i;
                        break;
                    }
                    let sum = summ.summary.items.iter().map(|(_, c)| c).sum::<usize>() as f32;
                    let max = *summ.summary.items.iter().map(|(_, c)| c).max().unwrap() as f32;
                    if 1.0 - max / sum < tau {
                        homogenity_depth = i;
                        break;
                    }
                }
                (path.len(), homogenity_depth)
            } else {
                (0, 0)
            }
        })
    }

    pub fn sample(&self) -> PyResult<(Py<PyArray1<f32>>, Option<PyObject>)> {
        let reader = self.writer.as_ref().unwrap().reader();
        let mut rng = SmallRng::from_entropy();
        let mut parent_addr = reader.root_address();

        while let Some(pat) = reader
            .get_node_plugin_and::<Dirichlet, _, _>(parent_addr, |p| p.marginal_sample(&mut rng))
            .unwrap()
        {
            parent_addr = pat;
        }
        let gil = pyo3::Python::acquire_gil();
        let py = gil.python();
        let vec = reader
            .get_node_plugin_and::<DiagGaussian, _, _>(parent_addr, |p| p.sample(&mut rng))
            .map(|m| {
                Array1::from_shape_vec((m.len(),), m)
                    .unwrap()
                    .into_pyarray(py)
                    .to_owned()
            })
            .unwrap();
        let dict = PyDict::new(py);
        let summ = match reader.get_node_label_summary(parent_addr) {
            Some(s) => {
                dict.set_item("errors", s.errors)?;
                dict.set_item("nones", s.nones)?;
                dict.set_item("items", s.summary.items.to_vec())?;
                Some(dict.into())
            }
            None => None,
        };
        Ok((vec, summ))
    }

    pub fn tracker(&self, size: u64) -> PyBayesCovertree {
        let writer = self.writer.as_ref().unwrap();

        PyBayesCovertree {
            hkl: BayesCovertree::new(size as usize, &writer.reader()),
            tree: writer.reader(),
        }
    }
}
