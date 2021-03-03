use pointcloud::*;
use goko::errors::GokoError;
use crate::core::*;

use serde::{Deserialize, Serialize};
//use std::convert::Infallible;

mod parameters;
mod path;
mod knn;

pub use parameters::*;
pub use path::*;
pub use knn::*;

//use crate::parser::{parse_body, ParserService};

/// How the server processes the request, under the hood.
pub(crate) trait Process<D: PointCloud> {
    type Response: Serialize;
    fn process(self, reader: &CoreReader<D>) -> Result<Self::Response, GokoError>;
}

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
    /// The catch-all for errors
    Unknown(String, u16),
}

/// The response one gets back from the core server loop.
#[derive(Deserialize, Serialize)]
pub enum GokoResponse {
    Parameters(ParametersResponse),
    Knn(KnnResponse),
    RoutingKnn(RoutingKnnResponse),
    Path(PathResponse),
    Unknown(String, u16),
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
