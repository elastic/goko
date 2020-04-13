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

//! A apache arrow inspired columnar meta-data storage.
//! This should probably be phased out in favor of Arrow

use super::DataSource;
use crate::datasources::DataMemmap;

use crate::errors::{ParsingError, PointCloudError};
use flate2::read::GzDecoder;
use indexmap::IndexMap;
use std::ffi::OsStr;
use std::fs::File;
use std::io::Read;
use std::path::PathBuf;

extern crate csv;
use self::csv::Reader;

mod numeric;
use numeric::*;
mod bools;
use bools::*;
mod vector;
use vector::*;
mod strings;
use strings::*;
mod list;
pub use list::MetadataList;
use list::*;
/// Value enums for managing the various data this crate supports
pub mod values;
use values::*;
mod summary;
pub use summary::*;

/// The schema defined by a user to build a more complex metadata system from a CSV
#[derive(Debug, Clone, Default)]
pub struct LabelScheme {
    name_column: String,
    schema: IndexMap<String, Value>,
}

impl LabelScheme {
    /// Creates a new blank schema
    pub fn new() -> LabelScheme {
        LabelScheme {
            name_column: "".to_string(),
            schema: IndexMap::new(),
        }
    }

    #[doc(hidden)]
    pub fn schema_json(&self) -> String {
        format!(
            "{{{}}}",
            self.schema
                .iter()
                .map(|(k, v)| format!("\"{}\":\"{}\"", k, v.value_type()))
                .collect::<Vec<String>>()
                .join(",")
        )
    }

    #[doc(hidden)]
    pub fn add_value(&mut self, key: String, value: Value) {
        self.schema.insert(key, value);
    }

    /// Add a name column.
    /// This will be interpreted as a string, and will be the accessor for the points
    pub fn add_name_column(&mut self, name: &str) {
        self.name_column = name.to_string();
    }

    /// Add a string column, with a given string. If the CSV header has that string, it will load from that column
    pub fn add_string(&mut self, key: String) {
        self.schema.insert(key, Value::String("".to_string()));
    }

    /// Add a bool column, with a given string. If the CSV header has that string, it will load from that column
    pub fn add_bool(&mut self, key: String) {
        self.schema.insert(key, Value::Bool(false));
    }

    /// Add a float column, with a given string. If the CSV header has that string, it will load from that column
    pub fn add_f32(&mut self, key: String) {
        self.schema.insert(key, Value::Number(Number::Real(0.0)));
    }

    /// Add a float column, with a given string. If the CSV header has that string, it will load from that column
    pub fn add_u32(&mut self, key: String) {
        self.schema.insert(key, Value::Number(Number::Natural(0)));
    }

    /// Add a float column, with a given string. If the CSV header has that string, it will load from that column
    pub fn add_i32(&mut self, key: String) {
        self.schema.insert(key, Value::Number(Number::Integer(0)));
    }

    /// If there is a column with natural numbers intended to be used as a class label, it creates a accessor for that
    pub fn add_vector(&mut self, name: String, dim: usize, dtype: &str) {
        let v = match dtype {
            "f32" | "Real" => Vector::Real(vec![0.0; dim]),
            "u32" | "Natural" => Vector::Natural(vec![0; dim]),
            "i32" | "Integer" => Vector::Integer(vec![0; dim]),
            &_ => panic!("Don't know what type you provided"),
        };
        self.schema.insert(name, Value::Vector(v));
    }

    #[doc(hidden)]
    pub fn empty(&self) -> MetadataList {
        let mut metalist = MetadataList::new();
        for (k, v) in self.schema.iter() {
            metalist.insert(k.clone(), v.blank_list());
        }
        metalist
    }

    /// Opens a file, and attemps to create an internal representation of the data with our type
    pub fn open(&self, path: &PathBuf) -> Result<MetadataList, PointCloudError> {
        println!(
            "Opening {:?} with extension {:?}",
            path,
            path.extension().unwrap()
        );
        match path.extension().and_then(OsStr::to_str) {
            Some("dat") => self.open_memmap(path),
            Some("csv") => self.open_csv(path),
            Some("gz") => self.open_csv(path),
            _ => panic!(
                "Please provide either a CSV or a memmaped dat file, not {:?}",
                path
            ),
        }
    }

    #[doc(hidden)]
    pub fn open_memmap(&self, path: &PathBuf) -> Result<MetadataList, PointCloudError> {
        assert!(self.schema.len() == 1);
        let (name, val_type) = self.schema.iter().next().unwrap();
        let labels_dim;
        if let Value::Vector(v) = val_type {
            labels_dim = v.len();
        } else {
            panic!("Need a vector only to use a MEMMAP file!");
        }

        let label = DataMemmap::new(labels_dim, &path).unwrap();
        let count = label.len();
        let mut label_in_ram = Vec::new();
        for i in 0..count {
            label_in_ram.extend_from_slice(label.get(i).unwrap());
        }
        let mut list = MetadataList::new();
        list.insert(name.clone(), VectorList::from_f32(label_in_ram, labels_dim));
        Ok(list)
    }

    #[doc(hidden)]
    pub fn open_csv(&self, path: &PathBuf) -> Result<MetadataList, PointCloudError> {
        if !path.exists() {
            panic!("CSV file {:?} does not exist", path);
        }
        println!("LabelScheme: {:?}", self);
        match File::open(&path) {
            Ok(file) => {
                if path.extension().unwrap() == "gz" {
                    self.read_csv(Reader::from_reader(GzDecoder::new(file)), path)
                } else {
                    self.read_csv(Reader::from_reader(file), path)
                }
            }
            Err(e) => panic!("Unable to open csv file {:#?}", e),
        }
    }

    fn read_csv<R: Read>(
        &self,
        mut rdr: Reader<R>,
        path: &PathBuf,
    ) -> Result<MetadataList, PointCloudError> {
        let mut count = 0;
        let mut internal_vals: Vec<(String, usize, ValueList)> = Vec::new();
        let mut names: IndexMap<usize, String> = IndexMap::new();
        let has_name: bool = self.name_column != "";
        let name_index: usize;
        // Get the y indexes
        {
            let columns = rdr.headers().expect("Can't read header.");
            for (val_name, value) in self.schema.iter() {
                let mut y_index = columns.len() + 1;
                for (i, c) in columns.iter().enumerate() {
                    if c == val_name {
                        y_index = i;
                    }
                }

                if y_index == columns.len() + 1 {
                    panic!("CSV has no {} column!", val_name);
                }
                internal_vals.push((val_name.clone(), y_index, value.blank_list()));
            }

            if has_name {
                let mut y_index = columns.len() + 1;
                for (i, c) in columns.iter().enumerate() {
                    if c == self.name_column {
                        y_index = i;
                    }
                }

                if y_index == columns.len() + 1 {
                    panic!("CSV has no {} column!", self.name_column);
                }
                name_index = y_index;
            } else {
                name_index = 0;
            }
        }

        for result in rdr.records() {
            // The iterator yields Result<StringRecord, Error>, so we check the
            // error here.
            let record = result.expect("Unable to read a record from the label CSV");
            for (val_name, val_name_index, list) in internal_vals.iter_mut() {
                match record.get(*val_name_index) {
                    Some(y) => {
                        if let Err(..) = list.read_csv_val(y) {
                            return Err(PointCloudError::ParsingError(
                                ParsingError::CSVReadError {
                                    file_name: path.to_string_lossy().to_string(),
                                    line_number: record.position().unwrap().line() as usize,
                                    key: val_name.clone(),
                                },
                            ));
                        }
                    }
                    None => {
                        return Err(PointCloudError::ParsingError(ParsingError::CSVReadError {
                            file_name: path.to_string_lossy().to_string(),
                            line_number: record.position().unwrap().line() as usize,
                            key: val_name.clone(),
                        }));
                    }
                }
            }
            if has_name {
                match record.get(name_index) {
                    Some(y) => {
                        names.insert(count, y.to_string());
                    }
                    None => {
                        return Err(PointCloudError::ParsingError(ParsingError::CSVReadError {
                            file_name: path.to_string_lossy().to_string(),
                            line_number: record.position().unwrap().line() as usize,
                            key: self.name_column.clone(),
                        }));
                    }
                }
            }
            count += 1;
        }

        let mut metalist = MetadataList::new();
        while let Some((name, _name_index, list)) = internal_vals.pop() {
            metalist.insert(name, list);
        }
        if has_name {
            metalist.insert_names(names);
        }
        Ok(metalist)
    }
    /*
    pub fn open_json(self,path: &PathBuf) -> MetadataList {

    }
    */
}
