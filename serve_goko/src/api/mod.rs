use goko::CoverTreeReader;
use http::Response;
use hyper::Body;
use pointcloud::*;
use goko::errors::GokoError;

use serde::{Deserialize, Serialize};
use std::ops::Deref;
//use std::convert::Infallible;

pub mod parameters;
pub mod path;
pub mod knn;

use parameters::*;
use path::*;
use knn::*;

//use crate::parser::{parse_body, ParserService};

/// How the server processes the request, under the hood.
pub(crate) trait Process<D: PointCloud> {
    type Response: Serialize;
    type Error;
    fn process(self, reader: &CoverTreeReader<D>) -> Result<Self::Response, Self::Error>;
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
    /// The catch all for errors
    Unknown(String, u16),
}

/// The response one gets back from the core server loop.
#[derive(Deserialize, Serialize)]
pub enum GokoResponse<N> {
    Parameters(ParametersResponse),
    Knn(KnnResponse<N>),
    RoutingKnn(RoutingKnnResponse<N>),
    Path(PathResponse<N>),
    Unknown(String, u16),
}

/// Response for KNN type queries, usually in a vec
#[derive(Deserialize, Serialize)]
pub struct NamedDistance<N> {
    /// The name of the point we're refering to
    pub name: N,
    /// Distance to that point
    pub distance: f32,
}

/// Response for queries that include distances to nodes, usually in a vec
#[derive(Deserialize, Serialize)]
pub struct NodeDistance<N> {
    /// The name of the center point of the node we're refering to
    pub name: N,
    /// The level the node is at
    pub layer: i32,
    /// The distance to the central node
    pub distance: f32,
}

/// Response when there is some kind of error
#[derive(Deserialize, Serialize)]
pub struct ErrorResponse {
    pub error: String,
}

impl<D: PointCloud, T: Deref<Target = D::Point> + Send + Sync> Process<D> for GokoRequest<T> {
    type Response = GokoResponse<D::Name>;
    type Error = GokoError;
    fn process(self, reader: &CoverTreeReader<D>) -> Result<Self::Response, Self::Error> {
        match self {
            GokoRequest::Parameters(p) => Ok(GokoResponse::Parameters(p.process(reader).unwrap())),
            GokoRequest::Knn(p) => p.process(reader).map(|p| GokoResponse::Knn(p)),
            GokoRequest::RoutingKnn(p) => p.process(reader).map(|p| GokoResponse::RoutingKnn(p)),
            GokoRequest::Path(p) => p.process(reader).map(|p| GokoResponse::Path(p)),
            GokoRequest::Unknown(response_string, status) => {
                Ok(GokoResponse::Unknown(response_string, status))
            }
        }
    }
}

fn parse_knn_query(uri: &Uri) -> usize {
    lazy_static! {
        static ref RE: Regex = Regex::new(r"k=(?P<k>\d+)").unwrap();
    }

    match uri.query().map(|s| RE.captures(s)).flatten() {
        Some(caps) => caps["k"].parse::<usize>().unwrap(),
        None => 10,
    }
}

pub async fn parse_http<P: PointParser>(request: Request<Body>, parser: &mut PointBuffer<P>) -> Result<GokoRequest<P::Point>, GokoClientError> {
    match (request.method(), request.uri().path()) {
        // Serve some instructions at /
        (&Method::GET, "/") => Ok(GokoRequest::Parameters(ParametersRequest)),
        (&Method::GET, "/knn") => {
            let k = parse_knn_query(request.uri());
            let point = parser.point(request).await?;
            Ok(GokoRequest::Knn(KnnRequest { point, k }))
        }
        (&Method::GET, "/routing_knn") => {
            let k = parse_knn_query(request.uri());
            let point = parser.point(request).await?;
            Ok(GokoRequest::RoutingKnn(RoutingKnnRequest { point, k }))

        }
        (&Method::GET, "/path") => {
            let point = parser.point(request).await?;
            Ok(GokoRequest::Path(PathRequest { point }))

        }
        // The 404 Not Found route...
        _ => Ok(GokoRequest::Unknown(String::new(), 404)),
    }
}

pub fn into_http<T>(response: GokoResponse<T>) -> Result<Response<Body>, GokoClientError> {
    let mut builder = http::response::Builder::new();
    let json_str = match response {
        GokoResponse::Parameters(p) => serde_json::to_string(&p).unwrap(),
        GokoResponse::Knn(p) => serde_json::to_string(&p).unwrap(),
        GokoResponse::RoutingKnn(p) => serde_json::to_string(&p).unwrap(),
        GokoResponse::Path(p) => serde_json::to_string(&p).unwrap(),
        GokoResponse::Unknown(response_string, status) => {
            builder = builder.status(status);
            response_string
        }
    }
    Ok(builder.body(Body::from(json_str)).unwrap())
}