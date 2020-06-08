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

use ndarray::{Array1};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::prelude::*;

use std::path::Path;
use std::sync::Arc;

use grandma::plugins::distributions::*;
use grandma::utils::*;
use grandma::*;
use pointcloud::*;

use crate::layer::*;
use crate::node::*;
use crate::plugins::*;

#[pyclass]
pub struct PyGrandma {
    builder: Option<CoverTreeBuilder>,
    temp_point_cloud: Option<PointCloud<L2>>,
    writer: Option<CoverTreeWriter<L2>>,
    reader: Option<Arc<CoverTreeReader<L2>>>,
    metric: String,
}

#[pymethods]
impl PyGrandma {
    #[new]
    fn new() -> PyResult<PyGrandma> {
        Ok(PyGrandma {
            builder: Some(CoverTreeBuilder::new()),
            temp_point_cloud: None,
            writer: None,
            reader: None,
            metric: "L2".to_string(),
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
    pub fn load_yaml_config(&mut self, file_name: String) -> PyResult<()> {
        let path = Path::new(&file_name);
        let (builder,point_cloud) = builder_from_yaml(&path).unwrap();
        self.builder = Some(builder);
        self.temp_point_cloud = Some(point_cloud);
        Ok(())
    }

    pub fn set_metric(&mut self, metric_name: String) {
        self.metric = metric_name;
    }

    pub fn fit(&mut self, data: Option<&PyArray2<f32>>, labels: Option<&PyArray2<f32>>) -> PyResult<()> {
        let point_cloud = if let Some(data) = data {
            let len = data.shape()[0];
            let data_dim = data.shape()[1];
            let labels_dim;
            let my_labels: Box<[f32]> = match labels {
                Some(labels) => {
                    labels_dim = labels.shape()[1];
                    Box::from(labels.as_slice().unwrap())
                }
                None => {
                    labels_dim = 1;
                    Box::from(vec![0.0; len])
                }
            };
            PointCloud::<L2>::simple_from_ram(
                Box::from(data.as_slice().unwrap()),
                data_dim,
                my_labels,
                labels_dim,
            )
            .unwrap()
        } else {
            if let Some(point_cloud) = self.temp_point_cloud.take() {
                point_cloud
            } else {
                panic!("No known point_cloud");
            }
        };
        
        println!("{:?}", point_cloud);
        let builder = self.builder.take();
        self.writer = Some(builder.unwrap().build(point_cloud).unwrap());
        let writer = self.writer.as_mut().unwrap();
        writer.add_plugin::<GrandmaDiagGaussian>(GrandmaDiagGaussian::singletons());
        writer.add_plugin::<GrandmaDirichlet>(DirichletTree {});
        let reader = writer.reader();

        self.reader = Some(Arc::new(reader));
        Ok(())
    }

    pub fn data_point(&self, point_index: u64) -> PyResult<Option<Py<PyArray1<f32>>>> {
        let reader = self.reader.as_ref().unwrap();
        let dim = reader.parameters().point_cloud.dim();
        Ok(
            match reader.parameters().point_cloud.get_point(point_index) {
                Err(_) => None,
                Ok(point) => {
                    let py_point = Array1::from_shape_vec((dim,), Vec::from(point)).unwrap();
                    let gil = GILGuard::acquire();
                    let py = gil.python();
                    Some(py_point.into_pyarray(py).to_owned())
                }
            },
        )
    }

    //pub fn layers(&self) ->
    pub fn top_scale(&self) -> Option<i32> {
        self.reader.as_ref().map(|r| r.scale_range().end - 1)
    }

    pub fn bottom_scale(&self) -> Option<i32> {
        self.reader.as_ref().map(|r| r.scale_range().start)
    }

    pub fn layers(&self) -> PyResult<IterLayers> {
        let reader = self.reader.as_ref().unwrap();
        let scale_indexes = reader.layers().map(|(si, _)| si).collect();
        Ok(IterLayers {
            parameters: Arc::clone(reader.parameters()),
            tree: reader.clone(),
            scale_indexes,
            index: 0,
        })
    }

    pub fn layer(&self, scale_index: i32) -> PyResult<PyGrandLayer> {
        let reader = self.reader.as_ref().unwrap();
        Ok(PyGrandLayer {
            parameters: Arc::clone(reader.parameters()),
            tree: reader.clone(),
            scale_index,
        })
    }

    pub fn node(&self, address: (i32, u64)) -> PyResult<PyGrandNode> {
        let reader = self.reader.as_ref().unwrap();
        // Check node exists
        reader.get_node_and(address, |_| true).unwrap();
        Ok(PyGrandNode {
            parameters: Arc::clone(reader.parameters()),
            address,
            tree: reader.clone(),
        })
    }

    pub fn root(&self) -> PyResult<PyGrandNode> {
        let reader = self.reader.as_ref().unwrap();
        self.node(reader.root_address())
    }

    pub fn knn(&self, point: &PyArray1<f32>, k: usize) -> Vec<(f32, u64)> {
        let results = self
            .reader
            .as_ref()
            .unwrap()
            .knn(point.as_slice().unwrap(), k)
            .unwrap();
        results
    }

    pub fn dry_insert(&self, point: &PyArray1<f32>) -> Vec<(f32, (i32, u64))> {
        let results = self
            .reader
            .as_ref()
            .unwrap()
            .dry_insert(point.as_slice().unwrap())
            .unwrap();
        results
    }

    pub fn kl_div_dirichlet(
        &self,
        prior_weight: f64,
        observation_weight: f64,
        size: u64,
    ) -> PyBayesCategoricalTracker {
        let reader = self.reader.as_ref().unwrap();
        let writer = self.writer.as_ref().unwrap();
        PyBayesCategoricalTracker {
            hkl: BayesCategoricalTracker::new(
                prior_weight,
                observation_weight,
                size as usize,
                writer.reader(),
            ),
            tree: Arc::clone(&reader),
        }
    }

    pub fn kl_div_dirichlet_basestats(
        &self,
        prior_weight: f64,
        observation_weight: f64,
        sequence_len: u64,
        num_sequences: u64,
        window_size: u64,
    ) -> Vec<Vec<PyKLDivergenceStats>> {
        let reader = self.writer.as_ref().unwrap().reader();
        let mut trainer = DirichletBaseline::new(reader);
        trainer.set_prior_weight(prior_weight);
        trainer.set_observation_weight(observation_weight);
        trainer.set_sequence_len(sequence_len as usize);
        trainer.set_num_sequences(num_sequences as usize);
        trainer.set_window_size(window_size as usize);
        trainer
            .train()
            .unwrap()
            .drain(0..)
            .map(|mut vstats| {
                vstats
                    .drain(0..)
                    .map(|stats| PyKLDivergenceStats { stats })
                    .collect()
            })
            .collect()
    }
}
