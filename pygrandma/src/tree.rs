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

use pyo3::prelude::*;

use numpy::{PyArray1, PyArray2};

use grandma::plugins::utils::*;
use grandma::plugins::*;
use grandma::*;
use pointcloud::*;
use std::sync::Arc;

use crate::layer::*;
use crate::node::*;
use crate::plugins::*;

#[pyclass]
pub struct PyGrandma {
    builder: Option<CoverTreeBuilder>,
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
    pub fn set_cutoff(&mut self, x: usize) {
        match &mut self.builder {
            Some(builder) => builder.set_cutoff(x),
            None => panic!("Set too late"),
        };
    }
    pub fn set_resolution(&mut self, x: i32) {
        match &mut self.builder {
            Some(builder) => builder.set_resolution(x),
            None => panic!("Set too late"),
        };
    }
    pub fn set_use_singletons(&mut self, x: bool) {
        match &mut self.builder {
            Some(builder) => builder.set_use_singletons(x),
            None => panic!("Set too late"),
        };
    }

    pub fn set_metric(&mut self, metric_name: String) {
        self.metric = metric_name;
    }

    pub fn fit(&mut self, data: &PyArray2<f32>, labels: Option<&PyArray2<f32>>) -> PyResult<()> {
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
        let pointcloud = PointCloud::<L2>::simple_from_ram(
            Box::from(data.as_slice().unwrap()),
            data_dim,
            my_labels,
            labels_dim,
        )
        .unwrap();
        println!("{:?}", pointcloud);
        let builder = self.builder.take();
        self.writer = Some(builder.unwrap().build(pointcloud).unwrap());
        let writer = self.writer.as_mut().unwrap();
        writer.add_plugin::<GrandmaDiagGaussian>(GrandmaDiagGaussian::singletons());
        writer.add_plugin::<GrandmaBucketProbs>(BucketProbsTree {});
        let reader = writer.reader();

        self.reader = Some(Arc::new(reader));
        Ok(())
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

    pub fn knn(&self, point: &PyArray1<f32>, k: usize) -> Vec<u64> {
        let results = self
            .reader
            .as_ref()
            .unwrap()
            .knn(point.as_slice().unwrap(), k)
            .unwrap();
        results.iter().map(|(_d, i)| *i).collect()
    }

    pub fn dry_insert(&self, point: &PyArray1<f32>) -> Vec<(i32, u64)> {
        let results = self
            .reader
            .as_ref()
            .unwrap()
            .dry_insert(point.as_slice().unwrap())
            .unwrap();
        results.iter().map(|(_, i)| *i).collect()
    }

    pub fn kl_div_frequentist(&self, k: Option<u64>) -> PyBucketHKLDivergence {
        let reader = self.reader.as_ref().unwrap();
        let writer = self.writer.as_ref().unwrap();
        let k = k.unwrap_or(0);
        PyBucketHKLDivergence {
            hkl: BucketHKLDivergence::new(k as usize, writer.reader()),
            tree: Arc::clone(&reader),
        }
    }
    pub fn kl_div_sgd(&self, learning_rate: f64, momentum: f64) -> PySGDHKLDivergence {
        let reader = self.reader.as_ref().unwrap();
        let writer = self.writer.as_ref().unwrap();
        PySGDHKLDivergence {
            hkl: SGDHKLDivergence::new(learning_rate, momentum, writer.reader()),
            tree: Arc::clone(&reader),
        }
    }
}
