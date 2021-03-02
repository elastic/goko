use goko::CoverTreeReader;
use pointcloud::*;

use serde::{Deserialize, Serialize};
use std::ops::Deref;

use goko::errors::GokoError;
use crate::core::*;
use super::{Process,NodeDistance};

/// Response: [`PathResponse`]
#[derive(Deserialize, Serialize)]
pub struct PathRequest<T> {
    pub point: T,
}

/// Request: [`PathRequest`]
#[derive(Deserialize, Serialize)]
pub struct PathResponse {
    pub path: Vec<NodeDistance>,
}

impl<D: PointCloud, T: Deref<Target = D::Point> + Send + Sync> Process<D> for PathRequest<T> {
    type Response = PathResponse;
    type Error = GokoError;
    fn process(self, reader: &CoreReader<D>) -> Result<Self::Response, Self::Error> {
        let knn = reader.tree.path(&self.point)?;
        let pc = &reader.tree.parameters().point_cloud;
        let resp: Result<Vec<NodeDistance>, GokoError> = knn
            .iter()
            .map(|(distance, (layer, pi))| {
                Ok(NodeDistance {
                    name: pc.name(*pi)?,
                    layer: *layer,
                    distance: *distance,
                })
            })
            .collect();
        Ok(PathResponse { path: resp? })
    }
}