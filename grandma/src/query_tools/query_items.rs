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

use crate::NodeAddress;
use pointcloud::PointIndex;
use std::cmp::Ordering::{self, Less};
use std::f32;

#[derive(Clone, Copy, Debug)]
pub(crate) struct QueryAddress {
    pub(crate) min_dist: f32,
    pub(crate) dist_to_center: f32,
    pub(crate) address: NodeAddress,
}

impl PartialEq for QueryAddress {
    fn eq(&self, other: &QueryAddress) -> bool {
        other.address == self.address
    }
}

impl Eq for QueryAddress {}

impl Ord for QueryAddress {
    fn cmp(&self, other: &QueryAddress) -> Ordering {
        self.partial_cmp(&other).unwrap_or(Ordering::Less)
    }
}

impl PartialOrd for QueryAddress {
    fn partial_cmp(&self, other: &QueryAddress) -> Option<Ordering> {
        // Backwards to make it a max heap.
        match other
            .min_dist
            .partial_cmp(&self.min_dist)
            .unwrap_or(Ordering::Equal)
        {
            Ordering::Greater => Some(Ordering::Greater),
            Ordering::Less => Some(Ordering::Less),
            Ordering::Equal => match other.address.0.cmp(&self.address.0) {
                Ordering::Greater => Some(Ordering::Greater),
                Ordering::Less => Some(Ordering::Less),
                Ordering::Equal => other.dist_to_center.partial_cmp(&self.dist_to_center),
            },
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct QuerySingleton {
    pub(crate) dist: f32,
    pub(crate) index: PointIndex,
}

impl QuerySingleton {
    pub(crate) fn new(index: PointIndex, dist: f32) -> QuerySingleton {
        QuerySingleton { dist, index }
    }
}

impl PartialEq for QuerySingleton {
    fn eq(&self, other: &QuerySingleton) -> bool {
        other.index == self.index
    }
}

impl Eq for QuerySingleton {}

impl Ord for QuerySingleton {
    fn cmp(&self, other: &QuerySingleton) -> Ordering {
        self.partial_cmp(&other).unwrap_or(Less)
    }
}

impl PartialOrd for QuerySingleton {
    fn partial_cmp(&self, other: &QuerySingleton) -> Option<Ordering> {
        self.dist.partial_cmp(&other.dist)
    }
}
