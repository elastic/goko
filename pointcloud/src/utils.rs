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

//! Utility data structures.

use crate::*;
use std::collections::HashMap;

/// The data structure for an adjacency matrix. This is a simple wrapper around a
/// hash-map whose keys are pairs of `PointIndexes` and whose values are the
/// distances between the two points. This is usually the method one want to access the data.
///
/// It assumes your data is symmetric and only stores the upper triangular matrix
#[derive(Debug)]
pub struct AdjMatrix {
    pub(crate) vals: HashMap<(PointIndex, PointIndex), f32>,
}

impl AdjMatrix {
    /// This gets by passing the smaller of the two indexes as the first element of
    /// the pair and the larger as the second.
    pub fn get(&self, i: PointIndex, j: PointIndex) -> Option<&f32> {
        if i < j {
            self.vals.get(&(i, j))
        } else {
            self.vals.get(&(j, i))
        }
    }

    /// Iterates over all distances and gets the minimum.
    pub fn min(&self) -> f32 {
        self.vals
            .iter()
            .fold(1.0 / 0.0, |a, (_k, v)| if v < &a { *v } else { a })
    }
}
