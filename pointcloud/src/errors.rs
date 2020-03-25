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

//! The errors that can occur when a point cloud is loading, working or saving
use std::error::Error;
use std::fmt;
use std::io;
use std::str;

///
pub type PointCloudResult<T> = Result<T, PointCloudError>;

/// Error type for the Point cloud
#[derive(Debug)]
pub enum PointCloudError {
    /// Unable to retrieve some data point (given by index) in a file (slice name)
    DataAccessError {
        /// Index of access error
        index: usize,
        /// File that had the access error
        slice_name: String,
    },
    /// Most common error, the given point name isn't present in the training data
    NameNotInTree(String),
    /// IO error when opening files
    IoError(io::Error),
    /// Parsing error when loading a CSV file
    ParsingError(ParsingError),
    ///
    NodeNestingError {
        /// Exact nesting error
        message: &'static str,
    },
}

impl fmt::Display for PointCloudError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            // not sure that cause should be included in message
            &PointCloudError::IoError(ref e) => write!(f, "{}", e),
            &PointCloudError::ParsingError(ref e) => write!(f, "{}", e),
            &PointCloudError::DataAccessError { .. } => {
                write!(f, "there was an issue grabbing a data point or label")
            }
            &PointCloudError::NameNotInTree { .. } => {
                write!(f, "there was an issue grabbing a name from the known names")
            }
            &PointCloudError::NodeNestingError { .. } => {
                write!(f, "There is a temporary node in a working tree")
            }
        }
    }
}

#[allow(deprecated)]
impl Error for PointCloudError {
    fn description(&self) -> &str {
        match self {
            // not sure that cause should be included in message
            &PointCloudError::IoError(ref e) => e.description(),
            &PointCloudError::ParsingError(ref e) => e.description(),
            &PointCloudError::DataAccessError { .. } => {
                "there was an issue grabbing a data point or label"
            }
            &PointCloudError::NameNotInTree { .. } => {
                "there was an issue grabbing a name from the known names"
            }
            &PointCloudError::NodeNestingError { .. } => {
                "There is a temporary node in a working tree"
            }
        }
    }

    fn cause(&self) -> Option<&dyn Error> {
        match self {
            &PointCloudError::IoError(ref e) => Some(e),
            &PointCloudError::ParsingError(ref e) => Some(e),
            &PointCloudError::DataAccessError { .. } => None,
            &PointCloudError::NameNotInTree { .. } => None,
            &PointCloudError::NodeNestingError { .. } => None,
        }
    }
}

impl From<io::Error> for PointCloudError {
    fn from(err: io::Error) -> Self {
        PointCloudError::IoError(err)
    }
}

impl From<PointCloudError> for io::Error {
    fn from(err: PointCloudError) -> Self {
        match err {
            PointCloudError::IoError(e) => e,
            e => io::Error::new(io::ErrorKind::Other, Box::new(e)),
        }
    }
}

impl PointCloudError {
    /// This error occurs when we try to doublenest a node, this should not occor
    pub fn node_nesting(message: &'static str) -> PointCloudError {
        PointCloudError::NodeNestingError { message }
    }

    /// If we can't get an element from a loaded data file, gives the i and filename
    pub fn data_access(index: usize, slice_name: String) -> PointCloudError {
        PointCloudError::DataAccessError { index, slice_name }
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
        fmt::Debug::fmt(self, f)
    }
}

impl Error for ParsingError {
    fn description(&self) -> &str {
        match self {
            // not sure that cause should be included in message
            &ParsingError::MalformedYamlError { .. } => "there is a error reading a yaml entry",
            &ParsingError::MissingYamlError { .. } => "not all message fields set",
            &ParsingError::CSVReadError { .. } => "issue reading a CSV entry",
            &ParsingError::RegularParsingError(..) => "Error parsing a string",
        }
    }

    fn cause(&self) -> Option<&dyn Error> {
        match self {
            &ParsingError::MalformedYamlError { .. } => None,
            &ParsingError::MissingYamlError { .. } => None,
            &ParsingError::CSVReadError { .. } => None,
            &ParsingError::RegularParsingError(..) => None,
        }
    }
}
