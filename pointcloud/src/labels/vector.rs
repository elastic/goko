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

use super::*;
use crate::errors::PointCloudError;
use serde::{Deserialize, Serialize};

/*
enum Vector {
    Real(Vec<f32>),
    Natural(Vec<u32>),
    Integer(Vec<i32>),
}
*/

fn get_row<T: Clone>(data: &[T], i: usize, dim: usize) -> Result<Vec<T>, PointCloudError> {
    match data.get(dim * i..(dim * i + dim)) {
        None => Err(PointCloudError::data_access(
            i,
            "Vector label access error".to_string(),
        )),
        Some(x) => Ok(x.to_vec()),
    }
}

fn get_set<T: Clone>(data: &[T], indexes: &[usize], dim: usize) -> Result<Vec<T>, PointCloudError> {
    let mut v: Vec<T> = Vec::new();
    for i in indexes {
        match data.get(dim * i..(dim * i + dim)) {
            None => {
                return Err(PointCloudError::data_access(
                    *i,
                    "Vector label access error".to_string(),
                ));
            }
            Some(x) => v.extend_from_slice(x),
        }
    }
    Ok(v)
}

#[derive(Debug, Clone)]
pub struct VectorList {
    data: Vector,
    dim: usize,
}

impl VectorList {
    pub fn get_dim(&self) -> usize {
        self.dim
    }
}

impl VectorList {
    pub(crate) fn from_f32(data: Vec<f32>, dim: usize) -> ValueList {
        ValueList::VectorList(VectorList {
            data: Vector::Real(data),
            dim,
        })
    }
    pub(crate) fn from_u32(data: Vec<u32>, dim: usize) -> ValueList {
        ValueList::VectorList(VectorList {
            data: Vector::Natural(data),
            dim,
        })
    }
    pub(crate) fn from_i32(data: Vec<i32>, dim: usize) -> ValueList {
        ValueList::VectorList(VectorList {
            data: Vector::Integer(data),
            dim,
        })
    }
}

impl InternalValueList for VectorList {
    fn empty() -> ValueList {
        ValueList::VectorList(VectorList {
            dim: 0,
            data: Vector::Real(Vec::new()),
        })
    }
    fn get_set(&self, is: &[usize]) -> Result<ValueList, PointCloudError> {
        match &self.data {
            Vector::Real(data) => Ok(ValueList::VectorList(VectorList {
                dim: self.dim,
                data: Vector::Real(get_set(data, is, self.dim)?),
            })),
            Vector::Natural(data) => Ok(ValueList::VectorList(VectorList {
                dim: self.dim,
                data: Vector::Natural(get_set(data, is, self.dim)?),
            })),
            Vector::Integer(data) => Ok(ValueList::VectorList(VectorList {
                dim: self.dim,
                data: Vector::Integer(get_set(data, is, self.dim)?),
            })),
        }
    }

    fn get_summary(&self, indexes: &[usize]) -> Result<ValueSummary, PointCloudError> {
        let mut sum_power1 = vec![0.0; self.dim];
        let mut sum_power2 = vec![0.0; self.dim];
        let count = indexes.len();
        let dim = self.dim;
        match &self.data {
            Vector::Real(data) => {
                for i in indexes {
                    match data.get(dim * i..(dim * i + dim)) {
                        None => {
                            return Err(PointCloudError::data_access(
                                *i,
                                "Vector label access error".to_string(),
                            ));
                        }
                        Some(x) => {
                            for (x_i, y_i) in sum_power1.iter_mut().zip(x) {
                                *x_i += *y_i as f32;
                            }
                            for (x_i, y_i) in sum_power2.iter_mut().zip(x) {
                                *x_i += (y_i * y_i) as f32;
                            }
                        }
                    };
                }
            }
            Vector::Natural(data) => {
                for i in indexes {
                    match data.get(dim * i..(dim * i + dim)) {
                        None => {
                            return Err(PointCloudError::data_access(
                                *i,
                                "Vector label access error".to_string(),
                            ));
                        }
                        Some(x) => {
                            for (x_i, y_i) in sum_power1.iter_mut().zip(x) {
                                *x_i += *y_i as f32;
                            }
                            for (x_i, y_i) in sum_power2.iter_mut().zip(x) {
                                *x_i += (y_i * y_i) as f32;
                            }
                        }
                    };
                }
            }
            Vector::Integer(data) => {
                for i in indexes {
                    match data.get(dim * i..(dim * i + dim)) {
                        None => {
                            return Err(PointCloudError::data_access(
                                *i,
                                "Vector label access error".to_string(),
                            ));
                        }
                        Some(x) => {
                            for (x_i, y_i) in sum_power1.iter_mut().zip(x) {
                                *x_i += *y_i as f32;
                            }
                            for (x_i, y_i) in sum_power2.iter_mut().zip(x) {
                                *x_i += (y_i * y_i) as f32;
                            }
                        }
                    };
                }
            }
        }
        Ok(ValueSummary::VectorSummary(VectorSummary {
            sum_power1,
            sum_power2,
            count,
        }))
    }

    fn get(&self, i: usize) -> Result<Value, PointCloudError> {
        match &self.data {
            Vector::Real(data) => Ok(Value::Vector(Vector::Real(get_row(data, i, self.dim)?))),
            Vector::Natural(data) => {
                Ok(Value::Vector(Vector::Natural(get_row(data, i, self.dim)?)))
            }
            Vector::Integer(data) => {
                Ok(Value::Vector(Vector::Integer(get_row(data, i, self.dim)?)))
            }
        }
    }

    fn push(&mut self, x: Value) {
        if let Value::Vector(mut x_vec) = x {
            if self.dim == 0 {
                match x_vec {
                    Vector::Real(x_vec) => self.data = Vector::Real(x_vec),
                    Vector::Natural(x_vec) => self.data = Vector::Natural(x_vec),
                    Vector::Integer(x_vec) => self.data = Vector::Integer(x_vec),
                }
            } else {
                match (&mut self.data, &mut x_vec) {
                    (Vector::Real(data), Vector::Real(x_vec)) => {
                        data.append(x_vec);
                    }
                    (Vector::Natural(data), Vector::Natural(x_vec)) => {
                        data.append(x_vec);
                    }
                    (Vector::Integer(data), Vector::Integer(x_vec)) => {
                        data.append(x_vec);
                    }
                    _ => {
                        panic!("Tried to push a natural number onto a real or something like that. Check your metadata!");
                    }
                }
            }
        }
    }
    fn len(&self) -> usize {
        if self.dim == 0 {
            0
        } else {
            self.data.len() / self.dim
        }
    }
    fn is_empty(&self) -> bool {
        if self.dim == 0 {
            true
        } else {
            self.data.is_empty()
        }
    }
}

/// The first 2 moments of the emprical distrbution that this is summarising. And the count!
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct VectorSummary {
    sum_power1: Vec<f32>,
    sum_power2: Vec<f32>,
    count: usize,
}

impl VectorSummary {
    /// Variance of the vector data that produced this summary
    pub fn variance(&self) -> Vec<f32> {
        if self.count > 0 {
            let mut var = Vec::with_capacity(self.sum_power1.len());
            for (s1, s2) in self.sum_power1.iter().zip(&self.sum_power2) {
                let moment1 = s1 / (self.count as f32);
                let moment2 = s2 / (self.count as f32);
                var.push(moment2 - moment1 * moment1)
            }
            var
        } else {
            vec![0.0; self.sum_power1.len()]
        }
    }
    /// Mean of the vector data that produced this summary
    pub fn mean(&self) -> Vec<f32> {
        if self.count > 0 {
            let mut mean = Vec::with_capacity(self.sum_power1.len());
            for s1 in self.sum_power1.iter() {
                mean.push(s1 / (self.count as f32))
            }
            mean
        } else {
            vec![0.0; self.sum_power1.len()]
        }
    }
    /// Count of the vector data that produced this summary
    pub fn count(&self) -> u64 {
        self.count as u64
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct VectorShortSummary {
    var: Vec<f32>,
    mean: Vec<f32>,
}

impl Summary for VectorSummary {
    fn to_json(&self) -> String {
        let vss = VectorShortSummary {
            var: self.variance(),
            mean: self.mean(),
        };

        serde_json::to_string(&vss).unwrap()
    }

    fn combine(summaries: &[&ValueSummary]) -> Result<ValueSummary, PointCloudError> {
        let mut sum_power1 = Vec::new();
        let mut sum_power2 = Vec::new();
        let mut count = 0;
        for vs in summaries {
            if let ValueSummary::VectorSummary(vs) = vs {
                if !vs.sum_power1.is_empty() {
                    if sum_power1.is_empty() {
                        sum_power1 = vs.sum_power1.clone();
                        sum_power2 = vs.sum_power2.clone();
                        count = vs.count;
                    } else {
                        for (x_i, y_i) in sum_power1.iter_mut().zip(&vs.sum_power1) {
                            *x_i += *y_i;
                        }
                        for (x_i, y_i) in sum_power2.iter_mut().zip(&vs.sum_power2) {
                            *x_i += *y_i;
                        }
                        count += vs.count;
                    }
                }
            } else {
                return Err(PointCloudError::data_access(
                    0,
                    "Non-vector summary passed to the vector summary combine".to_string(),
                ));
            }
        }

        Ok(ValueSummary::VectorSummary(VectorSummary {
            sum_power1,
            sum_power2,
            count,
        }))
    }

    fn add(&mut self, v: &Value) -> Result<(), PointCloudError> {
        if let Value::Vector(v) = v {
            match v {
                Vector::Real(v) => {
                    if self.sum_power1.is_empty() {
                        self.sum_power1 = v.clone();
                        self.sum_power2 = v.iter().map(|x| x * x).collect();
                    } else {
                        for (x_i, y_i) in self.sum_power1.iter_mut().zip(v) {
                            *x_i += (*y_i) as f32;
                        }
                        for (x_i, y_i) in self.sum_power2.iter_mut().zip(v) {
                            *x_i += ((*y_i) * (*y_i)) as f32;
                        }
                        self.count += 1;
                    }
                }
                Vector::Natural(v) => {
                    if self.sum_power1.is_empty() {
                        self.sum_power1 = v.iter().map(|x| *x as f32).collect();
                        self.sum_power2 = v.iter().map(|x| (*x * *x) as f32).collect();
                    } else {
                        for (x_i, y_i) in self.sum_power1.iter_mut().zip(v) {
                            *x_i += (*y_i) as f32;
                        }
                        for (x_i, y_i) in self.sum_power2.iter_mut().zip(v) {
                            *x_i += ((*y_i) * (*y_i)) as f32;
                        }
                        self.count += 1;
                    }
                }
                Vector::Integer(v) => {
                    if self.sum_power1.is_empty() {
                        self.sum_power1 = v.iter().map(|x| *x as f32).collect();
                        self.sum_power2 = v.iter().map(|x| (*x * *x) as f32).collect();
                    } else {
                        for (x_i, y_i) in self.sum_power1.iter_mut().zip(v) {
                            *x_i += (*y_i) as f32;
                        }
                        for (x_i, y_i) in self.sum_power2.iter_mut().zip(v) {
                            *x_i += ((*y_i) * (*y_i)) as f32;
                        }
                        self.count += 1;
                    }
                }
            }
            Ok(())
        } else {
            Err(PointCloudError::data_access(
                0,
                "Non-vector summary passed to the vector summary update".to_string(),
            ))
        }
    }
}
