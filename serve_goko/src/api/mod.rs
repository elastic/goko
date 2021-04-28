use std::ops::Deref;
use pointcloud::PointCloud;
use crate::errors::InternalServiceError;
use crate::core::CoreReader;

use serde::{Deserialize, Serialize};
//use std::convert::Infallible;

mod parameters;
mod path;
mod knn;
mod tracker;

pub use parameters::*;
pub use path::*;
pub use tracker::*;
pub use knn::*;

/// A summary for a small number of categories.
#[derive(Deserialize, Serialize)]
pub enum GokoRequest<T> {
    /// With the HTTP server, send a `GET` request to `/` for this.
    /// 
    /// Response: [`ParametersResponse`]
    Parameters(ParametersRequest),
    /// With the HTTP server, send a `GET` request to `/knn?k=5` with a set of features in the body for this query, 
    /// will return with the response with the nearest 5 routing nbrs. 
    /// 
    /// See the chosen body parser for how to encode the body.
    /// 
    /// Response: [`KnnResponse`]
    Knn(KnnRequest<T>),
    /// With the HTTP server, send a `GET` request to `/routing_knn?k=5` with a set of features in the body for this query, will return with the response with the nearest 5 routing nbrs. 
    /// 
    /// See the chosen body parser for how to encode the body.
    /// 
    /// Response: [`KnnResponse`]
    RoutingKnn(RoutingKnnRequest<T>),
    /// With the HTTP server, send a `GET` request to `/path` with a set of features in the body for this query, will return with the response the path to the node this point belongs to. 
    /// 
    /// See the chosen body parser for how to encode the body.
    /// 
    /// Response: [`PathResponse`]
    Path(PathRequest<T>),
    /// The queries to manipulate the trackers, all under /track/
    /// 
    /// See : [`TrackingRequest`]
    Tracking(TrackingRequest<T>),
    /// The catch-all for errors
    Unknown(String, u16),
}
#[derive(Deserialize, Serialize)]
pub struct TrackingRequest<T> {
    pub tracker_name: Option<String>,
    pub request: TrackingRequestChoice<T>,
}


#[derive(Deserialize, Serialize)]
pub enum TrackingRequestChoice<T> {
    /// Track a point, send a `POST` request to `/track/point?tracker_name=TRACKER_NAME` with a set of features in the body for this query. 
    /// Omit the `TRACKER_NAME` query to use the default. You
    /// 
    /// See the chosen body parser for how to encode the body.
    /// 
    /// Response: [`TrackPathResponse`]
    TrackPoint(TrackPointRequest<T>),
    /// Unsupported for HTTP
    /// 
    /// Response: [`TrackPathResponse`]
    TrackPath(TrackPathRequest),
    /// Add a tracker, send a `POST` request to `/track/add?window_size=WINDOW_SIZE&tracker_name=TRACKER_NAME` with a set of features in the body for this query. 
    /// Omit the `TRACKER_NAME` query to use the default.
    /// 
    /// Response: [`AddTrackerResponse`]
    AddTracker(AddTrackerRequest),
    /// Get the status of a tracker, send a `GET` request to `/track/stats?window_size=WINDOW_SIZE&tracker_name=TRACKER_NAME`.
    /// Omit the `TRACKER_NAME` query to use the default.
    /// 
    /// Response: [`CurrentStatsResponse`]
    CurrentStats(CurrentStatsRequest),
}

/// The response one gets back from the core server loop.
#[derive(Deserialize, Serialize)]
pub enum GokoResponse {
    Parameters(ParametersResponse),
    Knn(KnnResponse),
    RoutingKnn(RoutingKnnResponse),
    Path(PathResponse),
    Tracking(TrackingResponse),
    Unknown(String, u16),
}

#[derive(Deserialize, Serialize)]
pub enum TrackingResponse {
    TrackPath(TrackPathResponse),
    AddTracker(AddTrackerResponse),
    CurrentStats(CurrentStatsResponse),
    Unknown(Option<String>,Option<usize>),
}

/// Response for KNN type queries, usually in a vec
#[derive(Deserialize, Serialize)]
pub struct NamedDistance {
    /// The name of the point we're refering to
    pub name: String,
    /// Distance to that point
    pub distance: f32,
}

/// Response for queries that include distances to nodes, usually in a vec
#[derive(Deserialize, Serialize)]
pub struct NodeDistance {
    /// The name of the center point of the node we're refering to
    pub name: String,
    /// The level the node is at
    pub layer: i32,
    /// The distance to the central node
    pub distance: f32,
}

impl<D: PointCloud, P> CoreReader<D, P>
where P: Deref<Target = D::Point> + Send + Sync + 'static {
    pub async fn process(&mut self, request: GokoRequest<P>) -> Result<GokoResponse,InternalServiceError> {
        match request {
            GokoRequest::Parameters(p) => p.process(self).map(|p| GokoResponse::Parameters(p)).map_err(|e| e.into()),
            GokoRequest::Knn(p) => p.process(self).map(|p| GokoResponse::Knn(p)).map_err(|e| e.into()),
            GokoRequest::RoutingKnn(p) => p.process(self).map(|p| GokoResponse::RoutingKnn(p)).map_err(|e| e.into()),
            GokoRequest::Path(p) => p.process(self).map(|p| GokoResponse::Path(p)).map_err(|e| e.into()),
            GokoRequest::Unknown(response_string, status) => {
                Ok(GokoResponse::Unknown(response_string, status))
            },
            GokoRequest::Tracking(p) => {
                if let Some(tracker_name) = &p.tracker_name {
                    if let TrackingRequestChoice::AddTracker(_) = p.request {
                        self.trackers.write().await.entry(tracker_name.clone()).or_insert_with(|| TrackerWorker::operator(self.tree.clone()));
                    }
                    match self.trackers.read().await.get(tracker_name) {
                        Some(t) => t.message(p).await.map(|r| GokoResponse::Tracking(r)),
                        None => Ok(GokoResponse::Tracking(TrackingResponse::Unknown(Some(tracker_name.clone()), None))),
                    }
                } else {
                    self.main_tracker.message(p).await.map(|r| GokoResponse::Tracking(r))
                }
            }
        }
    }
}