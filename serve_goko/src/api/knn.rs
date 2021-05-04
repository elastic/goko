use pointcloud::*;
use crate::core::*;

use serde::{Deserialize, Serialize};
use std::ops::Deref;

use goko::errors::GokoError;

use super::NamedDistance;

/// Response: [`KnnResponse`]
#[derive(Deserialize, Serialize)]
pub struct KnnRequest<T> {
    pub k: usize,
    pub point: T,
}

/// Request: [`KnnRequest`]
#[derive(Deserialize, Serialize)]
pub struct KnnResponse {
    pub knn: Vec<NamedDistance>,
}

impl<T> KnnRequest<T> {
    pub fn process<D>(self, reader: &mut CoreReader<D,T>) -> Result<KnnResponse, GokoError> 
    where 
        D: PointCloud, 
        T: Deref<Target = D::Point> + Send + Sync,
    {
        let knn = reader.tree.knn(&self.point, self.k)?;
        let pc = &reader.tree.parameters().point_cloud;
        let resp: Result<Vec<NamedDistance>, GokoError> = knn
            .iter()
            .map(|(distance, pi)| {
                Ok(NamedDistance {
                    name: pc.name(*pi)?,
                    distance: *distance,
                })
            })
            .collect();

        Ok(KnnResponse { knn: resp? })
    }
}

/// Response: [`RoutingKnnResponse`]
#[derive(Deserialize, Serialize)]
pub struct RoutingKnnRequest<T> {
    pub k: usize,
    pub point: T,
}

/// Request: [`RoutingKnnRequest`]
#[derive(Deserialize, Serialize)]
pub struct RoutingKnnResponse {
    pub routing_knn: Vec<NamedDistance>,
}

impl<T> RoutingKnnRequest<T> {
    pub fn process<D>(self, reader: &CoreReader<D, T>) -> Result<RoutingKnnResponse, GokoError> 
    where 
        D: PointCloud, 
        T: Deref<Target = D::Point> + Send + Sync,
    {
        let knn = reader.tree.routing_knn(&self.point, self.k)?;
        let pc = &reader.tree.parameters().point_cloud;
        let resp: Result<Vec<NamedDistance>, GokoError> = knn
            .iter()
            .map(|(distance, pi)| {
                Ok(NamedDistance {
                    name: pc.name(*pi)?,
                    distance: *distance,
                })
            })
            .collect();

        Ok(RoutingKnnResponse { routing_knn: resp? })
    }
}