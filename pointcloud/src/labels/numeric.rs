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
use serde_json;

#[derive(Debug, Clone)]
pub struct NumberList {
    data: Vector,
}

impl NumberList {
    pub fn from_f32(data: Vec<f32>) -> NumberList {
        NumberList {
            data: Vector::Real(data),
        }
    }
    pub fn from_u32(data: Vec<u32>) -> NumberList {
        NumberList {
            data: Vector::Natural(data),
        }
    }
    pub fn from_i32(data: Vec<i32>) -> NumberList {
        NumberList {
            data: Vector::Integer(data),
        }
    }
}

impl InternalValueList for NumberList {
    fn new() -> ValueList {
        ValueList::NumberList(NumberList {
            data: Vector::Real(Vec::new()),
        })
    }
    fn get(&self, i: usize) -> Result<Value, PointCloudError> {
        match &self.data {
            Vector::Real(data) => Ok(Value::Number(Number::Real(data[i]))),
            Vector::Natural(data) => Ok(Value::Number(Number::Natural(data[i]))),
            Vector::Integer(data) => Ok(Value::Number(Number::Integer(data[i]))),
        }
    }
    fn get_set(&self, indexes: &[usize]) -> Result<ValueList, PointCloudError> {
        match &self.data {
            Vector::Real(data) => Ok(ValueList::NumberList(NumberList {
                data: Vector::Real(indexes.iter().map(|i| data[*i]).collect()),
            })),
            Vector::Natural(data) => Ok(ValueList::NumberList(NumberList {
                data: Vector::Natural(indexes.iter().map(|i| data[*i]).collect()),
            })),
            Vector::Integer(data) => Ok(ValueList::NumberList(NumberList {
                data: Vector::Integer(indexes.iter().map(|i| data[*i]).collect()),
            })),
        }
    }
    fn get_summary(&self, indexes: &[usize]) -> Result<ValueSummary, PointCloudError> {
        let mut sum_power1 = 0.0;
        let mut sum_power2 = 0.0;
        let count = indexes.len();
        match &self.data {
            Vector::Real(data) => {
                for i in indexes {
                    sum_power1 += data[*i] as f32;
                    sum_power2 += (data[*i] * data[*i]) as f32;
                }
            }
            Vector::Natural(data) => {
                for i in indexes {
                    sum_power1 += data[*i] as f32;
                    sum_power2 += (data[*i] * data[*i]) as f32;
                }
            }
            Vector::Integer(data) => {
                for i in indexes {
                    sum_power1 += data[*i] as f32;
                    sum_power2 += (data[*i] * data[*i]) as f32;
                }
            }
        }

        Ok(ValueSummary::NumberSummary(NumberSummary {
            sum_power1,
            sum_power2,
            count,
        }))
    }
    fn push(&mut self, x: Value) {
        if let Value::Number(x_val) = x {
            if self.data.len() == 0 {
                match x_val {
                    Number::Real(..) => self.data = Vector::Real(vec![]),
                    Number::Natural(..) => self.data = Vector::Natural(vec![]),
                    Number::Integer(..) => self.data = Vector::Integer(vec![]),
                }
            }
            match (&mut self.data, x_val) {
                (Vector::Real(data), Number::Real(val)) => data.push(val),
                (Vector::Natural(data), Number::Natural(val)) => data.push(val),
                (Vector::Integer(data), Number::Integer(val)) => data.push(val),

                _ => {
                    panic!("Tried to push a natural number onto a real or something like that. Check your metadata!");
                }
            }
        }
    }

    fn len(&self) -> usize {
        self.data.len()
    }
}

/// The first 2 moments of the emprical distrbution that this is summarising. And the count!
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct NumberSummary {
    sum_power1: f32,
    sum_power2: f32,
    count: usize,
}

impl NumberSummary {
    /// Variance of the numeric data that produced this summary
    pub fn variance(&self) -> f32 {
        if self.count > 0 {
            let moment1 = self.sum_power1 / (self.count as f32);
            let moment2 = self.sum_power2 / (self.count as f32);
            moment2 - moment1 * moment1
        } else {
            0.0
        }
    }
    /// Mean of the numeric data that produced this summary
    pub fn mean(&self) -> f32 {
        if self.count > 0 {
            self.sum_power1 / (self.count as f32)
        } else {
            0.0
        }
    }
    /// Count of the numeric data that produced this summary
    pub fn count(&self) -> u64 {
        self.count as u64
    }
}

#[derive(Debug, Deserialize, Serialize)]
struct NumericShortSummary {
    var: f32,
    mean: f32,
}

impl Summary for NumberSummary {
    fn to_json(&self) -> String {
        let vss = NumericShortSummary {
            var: self.variance(),
            mean: self.mean(),
        };

        serde_json::to_string(&vss).unwrap()
    }

    fn combine(summaries: &[&ValueSummary]) -> Result<ValueSummary, PointCloudError> {
        let mut sum_power1 = 0.0;
        let mut sum_power2 = 0.0;
        let mut count = 0;
        for vs in summaries {
            if let ValueSummary::NumberSummary(vs) = vs {
                sum_power1 += vs.sum_power1;
                sum_power2 += vs.sum_power2;
                count += vs.count;
            } else {
                return Err(PointCloudError::data_access(
                    0,
                    "Non-number summary passed to the number summary combine".to_string(),
                ));
            }
        }

        Ok(ValueSummary::NumberSummary(NumberSummary {
            sum_power1,
            sum_power2,
            count,
        }))
    }

    fn add(&mut self, v: &Value) -> Result<(), PointCloudError> {
        if let Value::Number(v) = v {
            match v {
                Number::Real(v) => {
                    self.sum_power1 += v;
                    self.sum_power2 += v * v;
                    self.count += 1;
                }
                Number::Natural(v) => {
                    self.sum_power1 += *v as f32;
                    self.sum_power2 += (v * v) as f32;
                    self.count += 1;
                }
                Number::Integer(v) => {
                    self.sum_power1 += *v as f32;
                    self.sum_power2 += (v * v) as f32;
                    self.count += 1;
                }
            }
            Ok(())
        } else {
            Err(PointCloudError::data_access(
                0,
                "Non-number summary passed to the number summary update".to_string(),
            ))
        }
    }
}
