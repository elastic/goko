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

//! This is intended to be the data structure for a summary of your metadata over some subset of indexes.

use super::*;
use crate::errors::PointCloudError;
/// Holder enum for all summaries
#[derive(Debug, Clone, Deserialize, Serialize)]
pub enum ValueSummary {
    #[doc(hidden)]
    Null,
    /// Boolean summary
    BoolSummary(BoolSummary),
    /// Numeric summary
    NumberSummary(NumberSummary),
    /// Vector summary
    VectorSummary(VectorSummary),
    /// String summary
    StringSummary(StringSummary),
}

/// The actual trait the columnar data sources has to implement
pub trait Summary {
    /// Adding a single value to the summary. When implementing please check that your value is compatible with your summary
    fn add(&mut self, v: &Value) -> Result<(), PointCloudError>;
    /// Merging several summaries of your data source together. This results in a summary of underlying column over
    /// the union of the indexes used to create the input summaries.
    fn combine(v: &[&ValueSummary]) -> Result<ValueSummary, PointCloudError>;
    /// Dumps this to a json value.
    fn to_json(&self) -> String;
}

impl Summary for ValueSummary {
    fn add(&mut self, v: &Value) -> Result<(), PointCloudError> {
        match (self, v) {
            (ValueSummary::BoolSummary(bl), Value::Bool(..)) => bl.add(v),
            (ValueSummary::NumberSummary(nl), Value::Number(..)) => nl.add(v),
            (ValueSummary::VectorSummary(vl), Value::Vector(..)) => vl.add(v),
            (ValueSummary::StringSummary(sl), Value::String(..)) => sl.add(v),
            _ => {
                panic!("Tried to add a natural number onto a real summary or something like that. Check your metadata!");
            }
        }
    }

    fn combine(v: &[&ValueSummary]) -> Result<ValueSummary, PointCloudError> {
        match v[0] {
            ValueSummary::BoolSummary(..) => BoolSummary::combine(v),
            ValueSummary::NumberSummary(..) => NumberSummary::combine(v),
            ValueSummary::VectorSummary(..) => VectorSummary::combine(v),
            ValueSummary::StringSummary(..) => StringSummary::combine(v),
            ValueSummary::Null => Err(PointCloudError::data_access(
                0,
                "Tried to merge Null summaries".to_string(),
            )),
        }
    }

    fn to_json(&self) -> String {
        match self {
            ValueSummary::BoolSummary(bl) => bl.to_json(),
            ValueSummary::NumberSummary(nl) => nl.to_json(),
            ValueSummary::VectorSummary(vl) => vl.to_json(),
            ValueSummary::StringSummary(sl) => sl.to_json(),
            _ => {
                panic!("Tried to add a natural number onto a real summary or something like that. Check your metadata!");
            }
        }
    }
}

/// A map that relates summaries to the keys that the values came from. This
#[derive(Debug, Clone, Deserialize, Serialize, Default)]
pub struct MetaSummary {
    /// the actual container
    pub summaries: IndexMap<String, ValueSummary>,
}

impl MetaSummary {
    #[doc(hidden)]
    pub fn new() -> MetaSummary {
        let summaries = IndexMap::new();
        MetaSummary { summaries }
    }

    #[doc(hidden)]
    pub fn insert(&mut self, key: String, value_sum: ValueSummary) {
        self.summaries.insert(key, value_sum);
    }
    #[doc(hidden)]
    pub fn add(&mut self, label: &Metadata) -> Result<(), PointCloudError> {
        for (k, v) in label.iter() {
            match self.summaries.get_mut(k) {
                Some(l) => l.add(v),
                None => Err(PointCloudError::data_access(
                    0,
                    "label access error, no key".to_string(),
                )),
            }?;
        }
        Ok(())
    }

    #[doc(hidden)]
    pub fn combine(metasummaries: &[MetaSummary]) -> Result<MetaSummary, PointCloudError> {
        let mut summary_vecs: IndexMap<String, Vec<&ValueSummary>> = IndexMap::new();
        for metasummary in metasummaries.iter() {
            for (k, v) in &metasummary.summaries {
                summary_vecs
                    .entry(k.to_string())
                    .or_insert(Vec::new())
                    .push(v);
            }
        }

        let mut summaries: IndexMap<String, ValueSummary> = IndexMap::new();

        for (k, v) in summary_vecs {
            summaries.insert(k, ValueSummary::combine(&v)?);
        }
        Ok(MetaSummary { summaries })
    }

    /// Easy getter, otherwise use the exposed index map
    pub fn get(&self, key: &str) -> Option<&ValueSummary> {
        self.summaries.get(key)
    }

    /// Encodes this into a compact json summary
    pub fn to_json(&self) -> String {
        let sum = self
            .summaries
            .iter()
            .map(|(n, v)| format!("\"{}\":{}", n, v.to_json()))
            .collect::<Vec<String>>()
            .join(",");
        format!("{{{}}}", sum)
    }
}
