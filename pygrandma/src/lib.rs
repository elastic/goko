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

use ndarray::{Array,Array2};
use numpy::{IntoPyArray, PyArray1,PyArray2};
use pyo3::PyIterProtocol;

use grandma::*;
use grandma::layer::*;
use pointcloud::*;
use std::sync::Arc;

use rayon::prelude::*;


#[pyclass(module = "pygrandma")]
pub struct PyGrandma {
    builder: Option<CoverTreeBuilder>,
    writer: Option<CoverTreeWriter<L2>>,
    reader: Option<CoverTreeReader<L2>>,
    metric: String,
}

#[pymethods]
impl PyGrandma {
    #[new]
    fn new(obj: &PyRawObject) -> PyResult<()> {
        obj.init(PyGrandma {
            builder: Some(CoverTreeBuilder::new()),
            writer: None,
            reader: None,
            metric: "L2".to_string(),
        });
        Ok(())
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

    pub fn set_metric(&mut self, metric_name:String) {
        self.metric = metric_name;
    }

    pub fn fit(&mut self,data:&PyArray2<f32>, labels:Option<&PyArray2<f32>>) -> PyResult<()>  {
        let len = data.shape()[0];
        let data_dim = data.shape()[1];
        let labels_dim;
        let my_labels: Box<[f32]> = match labels {
            Some(labels) => {
                labels_dim = labels.shape()[1];
                Box::from(labels.as_slice().unwrap())
            },
            None => {
                labels_dim = 1;
                Box::from(vec![0.0;len])
            },
        };
        let pointcloud = PointCloud::<L2>::simple_from_ram(Box::from(data.as_slice().unwrap()),data_dim,my_labels,labels_dim).unwrap();
        println!("{:?}", pointcloud);
        let builder = self.builder.take();
        self.writer = Some(builder.unwrap().build(pointcloud).unwrap());
        self.reader = Some(self.writer.as_ref().unwrap().reader());
        Ok(())
    }

    //pub fn layers(&self) -> 
    pub fn top_scale(&self) -> Option<i32> {
        self.reader.as_ref().map(|r| r.scale_range().end - 1)
    }

    pub fn bottom_scale(&self) -> Option<i32> {
        self.reader.as_ref().map(|r| r.scale_range().start)
    }

    pub fn layer(&self,scale_index:i32) -> PyResult<PyGrandLayer> {
        let reader = self.reader.as_ref().unwrap();
        Ok(PyGrandLayer {
            parameters: Arc::clone(reader.parameters()),
            layer: reader.layer(scale_index).reader(),
        })
    }

    pub fn layers(&self,scale_index:i32) -> PyResult<PyGrandLayer> {
        let reader = self.reader.as_ref().unwrap();
        Ok(PyGrandLayer {
            parameters: Arc::clone(reader.parameters()),
            layer: reader.layer(scale_index).reader(),
        })
    }

    pub fn knn(&self,point:&PyArray1<f32>,k:usize) -> Vec<u64> {
        let results = self.reader.as_ref().unwrap().knn(point.as_slice().unwrap(),k).unwrap();
        results.iter().map(|(d,i)| *i).collect()
    }
}

#[pyclass(module = "pygrandma")]
pub struct PyGrandLayer {
    parameters: Arc<CoverTreeParameters<L2>>,
    layer: CoverLayerReader,
}

#[pymethods]
impl PyGrandLayer {
    pub fn radius(&self) -> f32 {
        self.parameters.scale_base.powi(self.layer.scale_index())
    }
    pub fn scale_index(&self) -> i32 {
        self.layer.scale_index()
    }
    pub fn len(&self) -> usize {
        self.layer.node_count()
    }
    pub fn center_indexes(&self) -> Vec<u64> {
        self.layer.map_nodes(|pi,_n| *pi as u64)
    }
    pub fn child_addresses(&self,point_index:u64) -> Option<Vec<(i32,u64)>> {
        self.layer.get_node_children_and(&point_index,|nested_address,child_addresses| {let mut v = vec![nested_address]; v.extend(child_addresses); v})
    }
    pub fn singleton_indexes(&self,point_index:u64) -> Option<Vec<u64>> {
        self.layer.get_node_and(&point_index,|n| Vec::from(n.singletons()))
    }

    pub fn is_leaf(&self,point_index:u64) -> Option<bool> {
        self.layer.get_node_and(&point_index,|n| n.is_leaf())
    }

    pub fn centers(&self) -> PyResult<(Py<PyArray1<u64>>,Py<PyArray2<f32>>)> {
        let mut centers = Vec::with_capacity(self.layer.node_count()*self.parameters.point_cloud.dim());
        let mut centers_indexes = Vec::with_capacity(self.layer.node_count());
        self.layer.for_each_node(|pi,_n| {
            centers_indexes.push(*pi);
            centers.extend(self.parameters.point_cloud.get_point(*pi).unwrap());
        });
        let py_center_indexes = Array::from(centers_indexes);
        let py_centers = Array2::from_shape_vec((self.layer.node_count(), self.parameters.point_cloud.dim()), centers).unwrap();
        let gil = GILGuard::acquire();
        let py = gil.python();
        Ok((py_center_indexes.into_pyarray(py).to_owned(),py_centers.into_pyarray(py).to_owned()))
    }

    pub fn child_points(&self,point_index:u64) -> PyResult<Option<Py<PyArray2<f32>>>> {
        let dim = self.parameters.point_cloud.dim();
        Ok(self.layer.get_node_children_and(&point_index,|nested_address,child_addresses| {
            let count = child_addresses.len() + 1;
            let mut centers: Vec<f32> = Vec::with_capacity(count*dim);
            centers.extend(self.parameters.point_cloud.get_point(nested_address.1).unwrap());
            for na in child_addresses {
                centers.extend(self.parameters.point_cloud.get_point(na.1).unwrap());
            }
            let py_centers = Array2::from_shape_vec((count,dim), centers).unwrap();
            let gil = GILGuard::acquire();
            let py = gil.python();
            py_centers.into_pyarray(py).to_owned()
        }))
    }
    pub fn singleton_points(&self,point_index:u64) -> PyResult<Option<Py<PyArray2<f32>>>> {
        let dim = self.parameters.point_cloud.dim();
        Ok(self.layer.get_node_and(&point_index,|node| {
            let singletons = node.singletons();
            let mut centers: Vec<f32> = Vec::with_capacity(singletons.len()*dim);
            for pi in singletons {
                centers.extend(self.parameters.point_cloud.get_point(*pi).unwrap());
            }
            let py_centers = Array2::from_shape_vec((singletons.len(),dim), centers).unwrap();
            let gil = GILGuard::acquire();
            let py = gil.python();
            py_centers.into_pyarray(py).to_owned()
        }))
    }
}

#[pymodule]
fn pygrandma(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    m.add_class::<PyGrandma>()?;
    Ok(())
}

