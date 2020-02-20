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

#[derive(Debug, Clone)]
pub struct BoolList {
    data: Vec<bool>,
}

impl BoolList {
    pub fn from_bools(data: Vec<bool>) -> Result<BoolList, PointCloudError> {
        Ok(BoolList { data })
    }
}

impl InternalValueList for BoolList {
    fn new() -> ValueList {
        ValueList::BoolList(BoolList { data: Vec::new() })
    }
    fn get(&self, i: usize) -> Result<Value, PointCloudError> {
        Ok(Value::Bool(self.data[i]))
    }
    fn get_set(&self, indexes: &[usize]) -> Result<ValueList, PointCloudError> {
        Ok(ValueList::BoolList(BoolList {
            data: indexes.iter().map(|i| self.data[*i]).collect(),
        }))
    }

    fn get_summary(&self, indexes: &[usize]) -> Result<ValueSummary, PointCloudError> {
        let mut true_count = 0;
        let mut false_count = 0;
        for i in indexes {
            if self.data[*i] {
                true_count += 1;
            } else {
                false_count += 1;
            }
        }

        Ok(ValueSummary::BoolSummary(BoolSummary {
            true_count,
            false_count,
        }))
    }

    fn push(&mut self, x: Value) {
        if let Value::Bool(x) = x {
            self.data.push(x);
        }
    }
    fn len(&self) -> usize {
        self.data.len()
    }
}

/// The count of the trues and falses that underlay the producing data
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct BoolSummary {
    /// The true count
    pub true_count: usize,
    /// The false count
    pub false_count: usize,
}

impl Summary for BoolSummary {
    fn to_json(&self) -> String {
        format!("t:{},f:{}", self.true_count, self.false_count)
    }

    fn combine(summaries: &[&ValueSummary]) -> Result<ValueSummary, PointCloudError> {
        let mut true_count = 0;
        let mut false_count = 0;
        for vs in summaries {
            if let ValueSummary::BoolSummary(vs) = vs {
                true_count += vs.true_count;
                false_count += vs.false_count;
            } else {
                return Err(PointCloudError::data_access(
                    0,
                    "Non-boolean summary passed to the boolean summary combine".to_string(),
                ));
            }
        }

        Ok(ValueSummary::BoolSummary(BoolSummary {
            true_count,
            false_count,
        }))
    }

    fn add(&mut self, v: &Value) -> Result<(), PointCloudError> {
        if let Value::Bool(v) = v {
            if *v {
                self.true_count += 1;
            } else {
                self.false_count += 1;
            }
            Ok(())
        } else {
            Err(PointCloudError::data_access(
                0,
                "Non-boolean summary passed to the boolean summary update".to_string(),
            ))
        }
    }
}
