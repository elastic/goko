use pointcloud::*;

use serde::{Deserialize, Serialize};
use std::ops::Deref;

use super::NodeDistance;
use crate::core::*;
use goko::errors::GokoError;

/// Response: [`PathResponse`]
#[derive(Deserialize, Serialize)]
pub struct PathRequest<T> {
    pub point: T,
}

/// Request: [`PathRequest`]
#[derive(Deserialize, Serialize)]
pub struct PathResponse<L: Summary> {
    pub path: Vec<NodeDistance<L>>,
}

impl<T> PathRequest<T> {
    pub fn process<D>(
        self,
        reader: &mut CoreReader<D, T>,
    ) -> Result<PathResponse<D::LabelSummary>, GokoError>
    where
        D: PointCloud,
        T: Deref<Target = D::Point> + Send + Sync,
    {
        let knn = reader.tree.path(&self.point)?;
        let pc = &reader.tree.parameters().point_cloud;

        let resp: Result<Vec<NodeDistance<D::LabelSummary>>, GokoError> = knn
            .iter()
            .map(|(distance, na)| {
                let label_summary = reader.tree.get_node_label_summary(*na).map(|s| (*s).clone());
                Ok(NodeDistance {
                    name: pc.name(na.point_index())?,
                    layer: na.scale_index(),
                    distance: *distance,
                    label_summary,
                })
            })
            .collect();
        Ok(PathResponse { path: resp? })
    }
}
