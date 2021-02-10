use goko::CoverTreeReader;
use http::{Request, Response};
use hyper::{Body, Method};
use pointcloud::*;
use regex::Regex;

use lazy_static::lazy_static;

use serde::{Deserialize, Serialize};
use std::ops::Deref;

use goko::errors::GokoError;

pub mod parameters;
pub mod path;
pub mod knn;

use parameters::*;
use path::*;
use knn::*;

use crate::parser::{parse_body, ParserService};

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
            GokoRequest::Knn(p) => Ok(GokoResponse::Knn(p.process(reader)?)),
            GokoRequest::RoutingKnn(p) => Ok(GokoResponse::RoutingKnn(p.process(reader)?)),
            GokoRequest::Path(p) => Ok(GokoResponse::Path(p.process(reader)?)),
            GokoRequest::Unknown(response_string, status) => {
                Ok(GokoResponse::Unknown(response_string, status))
            }
        }
    }
}

pub(crate) async fn parse_request<P: ParserService>(
    parser: &P,
    request: Request<Body>,
) -> Result<GokoRequest<P::Point>, hyper::Error> {
    let (parts, body) = request.into_parts();
    match (parts.method, parts.uri.path()) {
        // Serve some instructions at /
        (Method::GET, "/") => Ok(GokoRequest::Parameters(ParametersRequest)),
        (Method::GET, "/knn") => {
            lazy_static! {
                static ref RE: Regex = Regex::new(r"k=(?P<k>\d+)").unwrap();
            }

            let k: usize = match parts.uri.query().map(|s| RE.captures(s)).flatten() {
                Some(caps) => caps["k"].parse::<usize>().unwrap(),
                None => 10,
            };
            let point = parse_body(parser, parts.headers.get("Content-Type"), body).await?;
            Ok(GokoRequest::Knn(KnnRequest { point, k }))
        }
        (Method::GET, "/routing_knn") => {
            lazy_static! {
                static ref RE: Regex = Regex::new(r"k=(?P<k>\d+)").unwrap();
            }

            let k: usize = match parts.uri.query().map(|s| RE.captures(s)).flatten() {
                Some(caps) => caps["k"].parse::<usize>().unwrap(),
                None => 10,
            };
            let point = parse_body(parser, parts.headers.get("Content-Type"), body).await?;
            Ok(GokoRequest::RoutingKnn(RoutingKnnRequest { point, k }))
        }
        (Method::GET, "/path") => {
            let point = parse_body(parser, parts.headers.get("Content-Type"), body).await?;
            Ok(GokoRequest::Path(PathRequest { point }))
        }
        // The 404 Not Found route...
        _ => Ok(GokoRequest::Unknown(String::new(), 404)),
    }
}

pub(crate) fn into_response<T: Serialize>(
    result: Result<GokoResponse<T>, GokoError>,
) -> Response<Body> {
    match result {
        Err(e) => {
            let error_detail = ErrorResponse {
                error: e.to_string(),
            };
            http::response::Builder::new()
                .status(500)
                .body(Body::from(serde_json::to_string(&error_detail).unwrap()))
                .unwrap()
        }
        Ok(resp) => {
            let mut builder = http::response::Builder::new();
            let json_str = match resp {
                GokoResponse::Parameters(p) => serde_json::to_string(&p).unwrap(),
                GokoResponse::Knn(p) => serde_json::to_string(&p).unwrap(),
                GokoResponse::RoutingKnn(p) => serde_json::to_string(&p).unwrap(),
                GokoResponse::Path(p) => serde_json::to_string(&p).unwrap(),
                GokoResponse::Unknown(response_string, status) => {
                    builder = builder.status(status);
                    response_string
                }
            };
            builder.body(Body::from(json_str)).unwrap()
        }
    }
}