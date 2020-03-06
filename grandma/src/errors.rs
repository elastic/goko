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

//! The errors that can occor when a cover tree is loading, working or saving.
//! Most errors are floated up from `PointCloud` as that's the i/o layer.

use pointcloud::errors::PointCloudError;
use protobuf::ProtobufError;
use std::error::Error;
use std::fmt;
use std::io;
use std::str;

/// Helper type for a call that could go wrong.
pub type GrandmaResult<T> = Result<T, GrandmaError>;

/// Error type for MalwareBrot. Mostly this is a wrapper around `PointCloudError`, as the data i/o where most errors happen.
#[derive(Debug)]
pub enum GrandmaError {
    /// Unable to retrieve some data point (given by index) in a file (slice name)
    PointCloudError(PointCloudError),
    /// Most common error, the given point name isn't present in the training data
    NameNotInTree(String),
    /// IO error when opening files
    IoError(io::Error),
    /// Parsing error when loading a CSV file
    ParsingError(ParsingError),
    /// Inserted a nested node into a node that already had a nested child
    DoubleNest,
    /// Inserted a node before you changed it from a leaf node into a normal node. Insert the nested child first.
    InsertBeforeNest,
}

impl fmt::Display for GrandmaError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            // not sure that cause should be included in message
            &GrandmaError::IoError(ref e) => write!(f, "{}", e),
            &GrandmaError::ParsingError(ref e) => write!(f, "{}", e),
            &GrandmaError::PointCloudError(ref e) => write!(f, "{}", e),
            &GrandmaError::NameNotInTree { .. } => {
                write!(f, "there was an issue grabbing a name from the known names")
            }
            &GrandmaError::DoubleNest => write!(
                f,
                "Inserted a nested node into a node that already had a nested child"
            ),
            &GrandmaError::InsertBeforeNest => write!(
                f,
                "Inserted a node into a node that does not have a nested child"
            ),
        }
    }
}

#[allow(deprecated)]
impl Error for GrandmaError {
    fn description(&self) -> &str {
        match self {
            // not sure that cause should be included in message
            &GrandmaError::IoError(ref e) => e.description(),
            &GrandmaError::ParsingError(ref e) => e.description(),
            &GrandmaError::PointCloudError(ref e) => e.description(),
            &GrandmaError::NameNotInTree { .. } => {
                "there was an issue grabbing a name from the known names"
            }
            &GrandmaError::DoubleNest => {
                "Inserted a nested node into a node that already had a nested child"
            }
            &GrandmaError::InsertBeforeNest => {
                "Inserted a node into a node that does not have a nested child"
            }
        }
    }

    fn cause(&self) -> Option<&dyn Error> {
        match self {
            &GrandmaError::IoError(ref e) => Some(e),
            &GrandmaError::ParsingError(ref e) => Some(e),
            &GrandmaError::PointCloudError(ref e) => Some(e),
            &GrandmaError::NameNotInTree { .. } => None,
            &GrandmaError::DoubleNest => None,
            &GrandmaError::InsertBeforeNest => None,
        }
    }
}

impl From<PointCloudError> for GrandmaError {
    fn from(err: PointCloudError) -> Self {
        GrandmaError::PointCloudError(err)
    }
}

impl From<io::Error> for GrandmaError {
    fn from(err: io::Error) -> Self {
        GrandmaError::IoError(err)
    }
}

impl From<ProtobufError> for GrandmaError {
    fn from(err: ProtobufError) -> Self {
        GrandmaError::ParsingError(ParsingError::ProtobufError(err))
    }
}

impl From<GrandmaError> for io::Error {
    fn from(err: GrandmaError) -> Self {
        match err {
            GrandmaError::IoError(e) => e,
            e => io::Error::new(io::ErrorKind::Other, Box::new(e)),
        }
    }
}

/// A parsing error occored while doing something with text
#[derive(Debug)]
pub enum ParsingError {
    /// Yaml was messed up
    MalformedYamlError {
        /// The file that was messed up
        file_name: String,
        /// The value that was messed up
        field: String,
    },
    /// A needed field was missing from the file.
    MissingYamlError {
        /// The file
        file_name: String,
        /// The missing field
        field: String,
    },
    /// Some protobuff error happened
    ProtobufError(ProtobufError),
    /// An error reading the CSV
    CSVReadError {
        /// The file that the error occored in
        file_name: String,
        /// The line that was messed up
        line_number: usize,
        /// The column name that was messed up
        key: String,
    },
    /// Something else happened parsing a string
    RegularParsingError(&'static str),
}

impl fmt::Display for ParsingError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            // not sure that cause should be included in message
            &ParsingError::ProtobufError(ref e) => write!(f, "{}", e),
            &ParsingError::MalformedYamlError { .. } => {
                write!(f, "there is a error reading a yaml entry")
            }
            &ParsingError::MissingYamlError { .. } => write!(f, "not all message fields set"),
            &ParsingError::CSVReadError { .. } => write!(f, "issue reading a CSV entry"),
            &ParsingError::RegularParsingError(..) => write!(f, "Error parsing a string"),
        }
    }
}

#[allow(deprecated)]
impl Error for ParsingError {
    fn description(&self) -> &str {
        match self {
            // not sure that cause should be included in message
            &ParsingError::ProtobufError(ref e) => e.description(),
            &ParsingError::MalformedYamlError { .. } => "there is a error reading a yaml entry",
            &ParsingError::MissingYamlError { .. } => "not all message fields set",
            &ParsingError::CSVReadError { .. } => "issue reading a CSV entry",
            &ParsingError::RegularParsingError(..) => "Error parsing a string",
        }
    }

    fn cause(&self) -> Option<&dyn Error> {
        match self {
            &ParsingError::ProtobufError(ref e) => Some(e),
            &ParsingError::MalformedYamlError { .. } => None,
            &ParsingError::MissingYamlError { .. } => None,
            &ParsingError::CSVReadError { .. } => None,
            &ParsingError::RegularParsingError(..) => None,
        }
    }
}
