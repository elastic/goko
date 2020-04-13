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
use std::collections::BinaryHeap;

#[derive(Debug, Clone)]
pub struct StringList {
    data: Vec<String>,
}

impl StringList {
    pub fn from_str(data: Vec<String>) -> Result<StringList, PointCloudError> {
        Ok(StringList { data })
    }
}

impl InternalValueList for StringList {
    fn empty() -> ValueList {
        ValueList::StringList(StringList { data: Vec::new() })
    }
    fn get(&self, i: usize) -> Result<Value, PointCloudError> {
        Ok(Value::String(self.data[i].clone()))
    }
    fn get_set(&self, indexes: &[usize]) -> Result<ValueList, PointCloudError> {
        Ok(ValueList::StringList(StringList {
            data: indexes.iter().map(|i| self.data[*i].clone()).collect(),
        }))
    }

    fn get_summary(&self, indexes: &[usize]) -> Result<ValueSummary, PointCloudError> {
        let mut unique_strings = IndexMap::new();
        for i in indexes {
            *unique_strings.entry(self.data[*i].clone()).or_insert(0) += 1;
        }
        Ok(ValueSummary::StringSummary(StringSummary {
            unique_strings,
        }))
    }

    fn push(&mut self, x: Value) {
        if let Value::String(x) = x {
            self.data.push(x);
        }
    }
    fn len(&self) -> usize {
        self.data.len()
    }
    fn is_empty(&self) -> bool {
        self.data.is_empty()
    }
}

/// The count of the unique strings that produced this summary
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct StringSummary {
    /// the actual container
    pub unique_strings: IndexMap<String, u32>,
}

impl Summary for StringSummary {
    fn to_json(&self) -> String {
        let mut min_heap: BinaryHeap<(i64, &str)> = BinaryHeap::with_capacity(10);

        for (k, v) in self.unique_strings.iter() {
            if min_heap.len() < 10 {
                min_heap.push((-(*v as i64), k));
            } else if min_heap.peek().unwrap().0 > -(*v as i64) {
                min_heap.pop();
                min_heap.push((-(*v as i64), k));
            }
        }

        let mut unique_strings: IndexMap<String, u32> = IndexMap::new();
        for (v, k) in min_heap {
            unique_strings.insert(k.to_string(), -v as u32);
        }
        serde_json::to_string(&unique_strings).unwrap()
    }

    fn combine(summaries: &[&ValueSummary]) -> Result<ValueSummary, PointCloudError> {
        let mut unique_strings = IndexMap::new();
        for vs in summaries {
            if let ValueSummary::StringSummary(vs) = vs {
                for (k, v) in vs.unique_strings.iter() {
                    *unique_strings.entry(k.to_string()).or_insert(0) += v;
                }
            } else {
                return Err(PointCloudError::data_access(
                    0,
                    "Non-string summary passed to the string summary combine".to_string(),
                ));
            }
        }

        Ok(ValueSummary::StringSummary(StringSummary {
            unique_strings,
        }))
    }

    fn add(&mut self, v: &Value) -> Result<(), PointCloudError> {
        if let Value::String(v) = v {
            *self.unique_strings.entry(v.to_string()).or_insert(0) += 1;
            Ok(())
        } else {
            Err(PointCloudError::data_access(
                0,
                "Non-string summary passed to the string summary update".to_string(),
            ))
        }
    }
}
