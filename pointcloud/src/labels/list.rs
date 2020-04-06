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
use crate::*;

#[derive(Debug, Clone)]
pub(crate) enum ValueList {
    Null,
    BoolList(BoolList),
    NumberList(NumberList),
    VectorList(VectorList),
    StringList(StringList),
}

pub(crate) trait InternalValueList {
    fn empty() -> ValueList;
    fn get(&self, i: usize) -> Result<Value, PointCloudError>;
    fn get_set(&self, i: &[usize]) -> Result<ValueList, PointCloudError>;
    fn get_summary(&self, i: &[usize]) -> Result<ValueSummary, PointCloudError>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool;
    fn push(&mut self, x: Value);
}

impl InternalValueList for ValueList {
    fn empty() -> ValueList {
        ValueList::Null
    }
    fn get(&self, i: usize) -> Result<Value, PointCloudError> {
        match self {
            ValueList::Null => Err(PointCloudError::data_access(
                i,
                "Null label access error".to_string(),
            )),
            ValueList::BoolList(bl) => bl.get(i),
            ValueList::NumberList(nl) => nl.get(i),
            ValueList::VectorList(vl) => vl.get(i),
            ValueList::StringList(sl) => sl.get(i),
        }
    }
    fn get_set(&self, i: &[usize]) -> Result<ValueList, PointCloudError> {
        match self {
            ValueList::Null => Err(PointCloudError::data_access(
                0,
                "Null label access error".to_string(),
            )),
            ValueList::BoolList(bl) => bl.get_set(i),
            ValueList::NumberList(nl) => nl.get_set(i),
            ValueList::VectorList(vl) => vl.get_set(i),
            ValueList::StringList(sl) => sl.get_set(i),
        }
    }
    fn get_summary(&self, i: &[usize]) -> Result<ValueSummary, PointCloudError> {
        match self {
            ValueList::Null => Err(PointCloudError::data_access(
                0,
                "Null label access error".to_string(),
            )),
            ValueList::BoolList(bl) => bl.get_summary(i),
            ValueList::NumberList(nl) => nl.get_summary(i),
            ValueList::VectorList(vl) => vl.get_summary(i),
            ValueList::StringList(sl) => sl.get_summary(i),
        }
    }
    fn len(&self) -> usize {
        match self {
            ValueList::Null => 0,
            ValueList::BoolList(bl) => bl.len(),
            ValueList::NumberList(nl) => nl.len(),
            ValueList::VectorList(vl) => vl.len(),
            ValueList::StringList(sl) => sl.len(),
        }
    }
    fn is_empty(&self) -> bool {
        match self {
            ValueList::Null => true,
            ValueList::BoolList(bl) => bl.is_empty(),
            ValueList::NumberList(nl) => nl.is_empty(),
            ValueList::VectorList(vl) => vl.is_empty(),
            ValueList::StringList(sl) => sl.is_empty(),
        }
    }
    fn push(&mut self, x: Value) {
        match self {
            ValueList::BoolList(bl) => bl.push(x),
            ValueList::NumberList(nl) => nl.push(x),
            ValueList::VectorList(vl) => vl.push(x),
            ValueList::StringList(sl) => sl.push(x),
            ValueList::Null => match x {
                Value::Null => {}
                Value::Bool(b) => {
                    *self = BoolList::empty();
                    self.push(Value::Bool(b));
                }
                Value::Number(n) => {
                    *self = NumberList::empty();
                    self.push(Value::Number(n));
                }
                Value::Vector(v) => {
                    *self = VectorList::empty();
                    self.push(Value::Vector(v));
                }
                Value::String(v) => {
                    *self = StringList::empty();
                    self.push(Value::String(v));
                }
            },
        }
    }
}

impl ValueList {
    pub(crate) fn read_csv_val(&mut self, val: &str) -> Result<(), PointCloudError> {
        match self {
            ValueList::BoolList(bl) => {
                let y_val = match val.parse::<bool>() {
                    Ok(y_val) => y_val,
                    Err(..) => {
                        return Err(PointCloudError::ParsingError(
                            ParsingError::RegularParsingError("Can't read bool"),
                        ))
                    }
                };
                bl.push(Value::Bool(y_val));
            }
            ValueList::NumberList(nl) => {
                let y_val = match val.parse::<f32>() {
                    Ok(y_val) => y_val,
                    Err(..) => {
                        return Err(PointCloudError::ParsingError(
                            ParsingError::RegularParsingError("can't read float"),
                        ))
                    }
                };
                nl.push(Value::Number(Number::Real(y_val)));
            }
            ValueList::VectorList(vl) => {
                let mut this_label: Vec<f32> = vec![0.0; vl.get_dim()];
                let y_val = match val.parse::<usize>() {
                    Ok(y_val) => y_val,
                    Err(..) => {
                        return Err(PointCloudError::ParsingError(
                            ParsingError::RegularParsingError("can't read usize"),
                        ))
                    }
                };
                if y_val > vl.get_dim() {
                    return Err(PointCloudError::ParsingError(
                        ParsingError::RegularParsingError("out of dim access"),
                    ));
                }
                this_label[y_val] = 1.0;
                vl.push(Value::Vector(Vector::Real(this_label)));
            }
            ValueList::StringList(nl) => {
                nl.push(Value::String(val.to_string()));
            }
            ValueList::Null => {
                return Err(PointCloudError::ParsingError(
                    ParsingError::RegularParsingError("can't determine type of value to read."),
                ))
            }
        }
        Ok(())
    }
}

/// This is a pair of `IndexMap`s. One that stores the name to index, the second that stores the
#[derive(Debug)]
pub struct MetadataList {
    pub(crate) names: IndexMap<usize, PointName>,
    pub(crate) lists: IndexMap<String, ValueList>,
    pub(crate) count: usize,
}

impl MetadataList {
    pub(crate) fn new() -> Self {
        MetadataList {
            names: IndexMap::new(),
            lists: IndexMap::new(),
            count: 0,
        }
    }

    pub(crate) fn simple_vec(labels: Box<[f32]>, labels_dim: usize) -> MetadataList {
        let mut list = MetadataList::new();
        list.insert(
            "y".to_string(),
            VectorList::from_f32(Vec::from(labels), labels_dim),
        );
        list
    }

    pub(crate) fn insert(&mut self, key: String, list: ValueList) {
        if self.count > 0 {
            assert!(self.count == list.len());
            self.lists.insert(key, list);
        } else {
            self.count = list.len();
            self.lists.insert(key, list);
        }
    }

    pub(crate) fn insert_names(&mut self, names: IndexMap<usize, PointName>) {
        if self.count > 0 {
            assert!(self.count == names.len());
            self.names = names;
        } else {
            self.count = names.len();
            self.names = names;
        }
    }

    pub(crate) fn get_name(&self, i: usize) -> Option<String> {
        if !self.names.is_empty() {
            Some(self.names[&i].clone())
        } else {
            None
        }
    }

    /// Grabs the correct value from each columnar data list and adds them to an index map, then returns it to the user.
    /// Errors out if the data was inaccessible
    pub fn get(&self, i: usize) -> Result<Metadata, PointCloudError> {
        let mut values = IndexMap::new();
        for (k, v) in &self.lists {
            values.insert(k.to_string(), v.get(i)?);
        }
        Ok(values)
    }

    /// Grabs the metadata for all elements of your subset and returns a smaller MetadataList.
    /// Errors out if the data was inaccessible
    pub fn get_set(&self, indexes: &[usize]) -> Result<MetadataList, PointCloudError> {
        let mut subset = MetadataList::new();
        for (k, v) in self.lists.iter().map(|(k, v)| (k, v.get_set(indexes))) {
            subset.insert(k.to_string(), v?);
        }
        Ok(subset)
    }

    /// Grabs the metadata for all elements of your subset and returns a summary of the underlying data.
    /// Errors out if the data was inaccessible
    pub fn get_summary(&self, indexes: &[usize]) -> Result<MetaSummary, PointCloudError> {
        let mut summaries = MetaSummary::new();
        for (k, v) in self.lists.iter().map(|(k, v)| (k, v.get_summary(indexes))) {
            summaries.insert(k.to_string(), v?);
        }
        Ok(summaries)
    }

    /// Appends a the values contained in a metadata list to our metadata.
    /// Errors out if you are missing a key that the map has.
    pub fn push(
        &mut self,
        name: Option<PointName>,
        label: Metadata,
    ) -> Result<(), PointCloudError> {
        if let Some(n) = name {
            self.names.insert(self.count, n);
        }
        for (k, list) in self.lists.iter_mut() {
            match label.get(k) {
                Some(val) => list.push(val.clone()),
                None => {
                    return Err(PointCloudError::data_access(
                        0,
                        "label access error, no key".to_string(),
                    ));
                }
            };
        }
        self.count += 1;
        Ok(())
    }

    /// Outputs the scheme of the metadata. This is useful for creating new metadata object off of other data.
    pub fn scheme(&self) -> Result<LabelScheme, PointCloudError> {
        let mut deser = LabelScheme::new();
        for (k, l) in &self.lists {
            deser.add_value(k.to_string(), l.get(0)?);
        }
        Ok(deser)
    }

    ///
    pub fn keys(&self) -> Vec<String> {
        self.lists.keys().cloned().collect()
    }

    ///
    pub fn len(&self) -> usize {
        self.count
    }
    ///
    pub fn is_empty(&self) -> bool {
        self.count > 0
    }
}
