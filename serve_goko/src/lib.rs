use goko::CoverTreeReader;
use http::{Request, Response, StatusCode};
use hyper::{Body, Method};
use pointcloud::*;
use regex::Regex;

use lazy_static::lazy_static;

use serde::{Deserialize, Serialize};
use std::ops::Deref;

mod client;
pub use client::*;
mod parser;
pub use parser::*;
mod errors;
pub use errors::*;

pub async fn parse_request<P: ParserService>(
    parser: &P,
    request: Request<Body>,
) -> Result<GokoRequest<P::Point>, hyper::Error> {
    let (parts, body) = request.into_parts();
    match (parts.method, parts.uri.path()) {
        // Serve some instructions at /
        (Method::GET, "/") => Ok(GokoRequest::Parameters),
        (Method::GET, "/knn") => {
            lazy_static! {
                static ref RE: Regex = Regex::new(r"k=(?P<k>\d+)").unwrap();
            }

            let k: usize = match parts.uri.query().map(|s| RE.captures(s)).flatten() {
                Some(caps) => caps["k"].parse::<usize>().unwrap(),
                None => 10,
            };
            let vec = parse_body(parser, parts.headers.get("Content-Type"), body).await?;
            Ok(GokoRequest::Knn {
                vec,
                k,
            })
        }
        (Method::GET, "/routing_knn") => {
            lazy_static! {
                static ref RE: Regex = Regex::new(r"k=(?P<k>\d+)").unwrap();
            }

            let k: usize = match parts.uri.query().map(|s| RE.captures(s)).flatten() {
                Some(caps) => caps["k"].parse::<usize>().unwrap(),
                None => 10,
            };
            let vec = parse_body(parser, parts.headers.get("Content-Type"), body).await?;
            Ok(GokoRequest::RoutingKnn {
                vec,
                k,
            })
        }
        (Method::GET, "/path") => {
            let vec = parse_body(parser, parts.headers.get("Content-Type"), body).await?;
            Ok(GokoRequest::Path {
                vec,
            })
        }
        // The 404 Not Found route...
        _ => Ok(GokoRequest::Unknown),
    }
}

pub fn process<'a, D: PointCloud, T: Deref<Target = D::Point> + Send + Sync>(request: GokoRequest<T>, reader: &CoverTreeReader<D>) -> Response<Body> {
    use GokoRequest::*;
    match request {
        Parameters => {
            let params = reader.parameters();
            Response::new(Body::from(format!(
                "{{\"scale_base\":{},\"leaf_cutoff\":{},\"min_res_index\":{}}}",
                params.scale_base, params.leaf_cutoff, params.min_res_index
            )))
        }
        Knn {
            vec,
            k,
        } => {
            match reader.knn(&vec,k) {
                Ok(knn) => {
                    Response::new(Body::from(serde_json::to_string(&knn).unwrap()))
                }
                Err(e) => {
                    Response::new(Body::from(format!("{:?}",e)))
                }
            }
            
        }
        RoutingKnn {
            vec,
            k,
        } => {
            match reader.routing_knn(&vec,k) {
                Ok(knn) => {
                    Response::new(Body::from(serde_json::to_string(&knn).unwrap()))
                }
                Err(e) => {
                    Response::new(Body::from(format!("{:?}",e)))
                }
            }
            
        }
        Path {
            vec,
        } => {
            match reader.path(&vec) {
                Ok(knn) => {
                    Response::new(Body::from(serde_json::to_string(&knn).unwrap()))
                }
                Err(e) => {
                    Response::new(Body::from(format!("{:?}",e)))
                }
            }
            
        }
        Unknown => {
            let mut response = Response::new(Body::empty());
            *response.status_mut() = StatusCode::NOT_FOUND;
            response
        }
    }
}

/// A summary for a small number of categories.
#[derive(Deserialize, Serialize)]
pub enum GokoRequest<T> {
    Parameters,
    Knn {
        k: usize,
        vec: T,
    },
    RoutingKnn {
        k: usize,
        vec: T,
    },
    Path {
        vec: T,
    },
    Unknown,
}

