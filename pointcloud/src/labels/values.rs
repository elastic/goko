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

/// Number type
#[derive(Debug, Clone)]
pub enum Number {
    /// Real AKA float
    Real(f32),
    /// Natural AKA uint
    Natural(u32),
    /// Integer AKA int
    Integer(i32),
}
/// Vector type
#[derive(Debug, Clone)]
pub enum Vector {
    /// Real AKA float
    Real(Vec<f32>),
    /// Natural AKA uint
    Natural(Vec<u32>),
    /// Integer AKA int
    Integer(Vec<i32>),
}

impl Vector {
    /// returns the lenght of a typed vector
    pub fn len(&self) -> usize {
        match self {
            Vector::Real(x_vec) => x_vec.len(),
            Vector::Natural(x_vec) => x_vec.len(),
            Vector::Integer(x_vec) => x_vec.len(),
        }
    }
    ///
    pub fn is_empty(&self) -> bool {
        match self {
            Vector::Real(x_vec) => x_vec.is_empty(),
            Vector::Natural(x_vec) => x_vec.is_empty(),
            Vector::Integer(x_vec) => x_vec.is_empty(),
        }
    }
}

/// A value
#[derive(Debug, Clone)]
pub enum Value {
    /// Don't use this
    Null,
    /// Boolean type
    Bool(bool),
    /// Numeric type
    Number(Number),
    /// Vector type, typically one hot
    Vector(Vector),
    /// String type
    String(String),
}

impl Value {
    #[doc(hidden)]
    pub(crate) fn blank_list(&self) -> ValueList {
        match self {
            Value::Null => ValueList::empty(),
            Value::Bool(..) => BoolList::empty(),
            Value::Number(..) => NumberList::empty(),
            Value::String(..) => StringList::empty(),
            Value::Vector(v) => match v {
                Vector::Real(v) => VectorList::from_f32(vec![], v.len()),
                Vector::Natural(v) => VectorList::from_u32(vec![], v.len()),
                Vector::Integer(v) => VectorList::from_i32(vec![], v.len()),
            },
        }
    }
    /// String name of the value
    pub fn value_type(&self) -> &str {
        match self {
            Value::Null => "Null",
            Value::Bool(..) => "Bool",
            Value::Number(..) => "Numeric",
            Value::String(..) => "String",
            Value::Vector(..) => "Vector",
        }
    }
}

/// Basically a json entry with values being the supported values
pub type Metadata = IndexMap<String, Value>;
