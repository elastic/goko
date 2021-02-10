use goko::CoverTreeReader;
use pointcloud::*;

use serde::{Deserialize, Serialize};
use std::ops::Deref;

use goko::errors::GokoError;

use super::{Process,NamedDistance};

/// Response: [`KnnResponse`]
#[derive(Deserialize, Serialize)]
pub struct KnnRequest<T> {
    pub k: usize,
    pub point: T,
}

/// Request: [`KnnRequest`]
#[derive(Deserialize, Serialize)]
pub struct KnnResponse<N> {
    pub knn: Vec<NamedDistance<N>>,
}

impl<D: PointCloud, T: Deref<Target = D::Point> + Send + Sync> Process<D> for KnnRequest<T> {
    type Response = KnnResponse<D::Name>;
    type Error = GokoError;
    fn process(self, reader: &CoverTreeReader<D>) -> Result<Self::Response, Self::Error> {
        let knn = reader.knn(&self.point, self.k)?;
        let pc = &reader.parameters().point_cloud;
        let resp: Result<Vec<NamedDistance<D::Name>>, GokoError> = knn
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
pub struct RoutingKnnResponse<N> {
    pub routing_knn: Vec<NamedDistance<N>>,
}

impl<D: PointCloud, T: Deref<Target = D::Point> + Send + Sync> Process<D> for RoutingKnnRequest<T> {
    type Response = RoutingKnnResponse<D::Name>;
    type Error = GokoError;
    fn process(self, reader: &CoverTreeReader<D>) -> Result<Self::Response, Self::Error> {
        let knn = reader.routing_knn(&self.point, self.k)?;
        let pc = &reader.parameters().point_cloud;
        let resp: Result<Vec<NamedDistance<D::Name>>, GokoError> = knn
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