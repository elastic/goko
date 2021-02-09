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

use pointcloud::pc_errors::PointCloudError;
use protobuf::ProtobufError;
use std::error::Error;
use std::fmt;
use std::io;
use std::str;

/// Helper type for a call that could go wrong.
pub type GokoResult<T> = Result<T, GokoError>;

/// Error type for MalwareBrot. Mostly this is a wrapper around `PointCloudError`, as the data i/o where most errors happen.
#[derive(Debug)]
pub enum GokoError {
    /// Unable to retrieve some data point (given by index) in a file (slice name)
    PointCloudError(PointCloudError),
    /// Most common error, the given point name isn't present in the training data
    IndexNotInTree(usize),
    /// Parsing error when loading a CSV file
    ProtobufError(ProtobufError),
    /// Parsing error when loading a CSV file
    IoError(io::Error),
    /// The probability distribution you are trying to sample from is invalid, probably because it was infered from 0 points.
    InvalidProbDistro,
    /// Inserted a nested node into a node that already had a nested child
    DoubleNest,
    /// Inserted a node before you changed it from a leaf node into a normal node. Insert the nested child first.
    InsertBeforeNest,
}

impl fmt::Display for GokoError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match *self {
            GokoError::PointCloudError(ref e) => write!(f, "{}", e),
            GokoError::ProtobufError(ref e) => write!(f, "{}", e),
            GokoError::IoError(ref e) => write!(f, "{}", e),
            GokoError::IndexNotInTree { .. } => {
                write!(f, "there was an issue grabbing a name from the known names")
            }
            GokoError::DoubleNest => write!(
                f,
                "Inserted a nested node into a node that already had a nested child"
            ),
            GokoError::InvalidProbDistro => write!(
                f,
                "The probability distribution you are trying to sample from is invalid, probably because it was infered from 0 points."
            ),
            GokoError::InsertBeforeNest => write!(
                f,
                "Inserted a node into a node that does not have a nested child"
            ),
        }
    }
}

#[allow(deprecated)]
impl Error for GokoError {
    fn description(&self) -> &str {
        match *self {
            GokoError::PointCloudError(ref e) => e.description(),
            GokoError::ProtobufError(ref e) => e.description(),
            GokoError::IoError(ref e) => e.description(),
            GokoError::IndexNotInTree { .. } => {
                "there was an issue grabbing a name from the known names"
            }
            GokoError::DoubleNest => {
                "Inserted a nested node into a node that already had a nested child"
            }
            GokoError::InsertBeforeNest => {
                "Inserted a node into a node that does not have a nested child"
            }
            GokoError::InvalidProbDistro => {
                "The probability distribution you are trying to sample from is invalid, probably because it was infered from 0 points."
            }
        }
    }

    fn cause(&self) -> Option<&dyn Error> {
        match *self {
            GokoError::PointCloudError(ref e) => Some(e),
            GokoError::ProtobufError(ref e) => Some(e),
            GokoError::IoError(ref e) => Some(e),
            GokoError::IndexNotInTree { .. } => None,
            GokoError::DoubleNest => None,
            GokoError::InsertBeforeNest => None,
            GokoError::InvalidProbDistro => None,
        }
    }
}

impl From<PointCloudError> for GokoError {
    fn from(err: PointCloudError) -> Self {
        GokoError::PointCloudError(err)
    }
}

impl From<ProtobufError> for GokoError {
    fn from(err: ProtobufError) -> Self {
        GokoError::ProtobufError(err)
    }
}

impl From<io::Error> for GokoError {
    fn from(err: io::Error) -> Self {
        GokoError::IoError(err)
    }
}