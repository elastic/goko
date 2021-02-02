#![deny(warnings)]
use futures::{future, IntoFuture};
use hyper::rt::{Future, Stream};
use hyper::service::Service;
use hyper::{Body, Method, Request, Response, Server, StatusCode};

extern crate malwarebrot;
use malwarebrot::errors::*;
use malwarebrot::{CoverTree, PointCloud};

extern crate regex;
use regex::Regex;

//use std::io;
use std::path::Path;
use std::sync::Arc;
#[macro_use]
extern crate lazy_static;

use flate2::read::{DeflateDecoder,ZlibDecoder};
use rmp::decode;
use rmp::decode::ValueReadError;
use std::io::prelude::*;

/// We need to return different futures depending on the route matched,
/// and we can do that with an enum, such as `futures::Either`, or with
/// trait objects.
///
/// A boxed Future (trait object) is used as it is easier to understand
/// and extend with more types. Advanced users could switch to `Either`.
type BoxFut = Box<dyn Future<Item = Response<Body>, Error = hyper::Error> + Send>;
#[derive(Debug)]
struct CoverTreeService {
    ct: Arc<CoverTree>,
}

fn named_trace_response(ct: Arc<CoverTree>, sha256: String) -> Response<Body> {
    let mut response = Response::new(Body::empty());
    match ct.named_insert_trace(&sha256) {
        Ok(trace) => {
            let trace_report: String = trace
                .iter()
                .map(|node| node.report_json())
                .collect::<Vec<String>>()
                .join(",");
            *response.body_mut() = Body::from(format!("[{}]", trace_report));
        }
        Err(e) => match e {
            MalwareBrotError::NameNotInTree(_) => {
                *response.body_mut() = Body::from(format!("{:?}", e));
            }
            _ => {
                *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                *response.body_mut() = Body::from(format!("Failed on {:?}: {:?}", &sha256, e));
            }
        },
    };
    response
}

fn trace_response(ct: Arc<CoverTree>, features: Vec<f32>) -> Response<Body> {
    let mut response = Response::new(Body::empty());
    match ct.insert_trace(&features) {
        Ok(trace) => {
            let trace_report: String = trace
                .iter()
                .map(|node| node.report_json())
                .collect::<Vec<String>>()
                .join(",");

            *response.body_mut() = Body::from(format!("[{}]", trace_report));
        }
        Err(e) => {
            *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
            *response.body_mut() = Body::from(format!("Failed on trace with: {:?}", e));
        }
    };
    response
}

fn named_aprox_knn_response(ct: Arc<CoverTree>, sha256: String, k: usize) -> Response<Body> {
    let mut response = Response::new(Body::empty());
    match ct.named_center_knn_query(&sha256, k) {
        Ok(trace) => {
            let tr = serde_json::to_string(&trace).unwrap();
            *response.body_mut() = Body::from(tr);
        }
        Err(e) => match e {
            MalwareBrotError::NameNotInTree(_) => {
                *response.body_mut() = Body::from(format!("{:?}", e));
            }
            _ => {
                *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                *response.body_mut() = Body::from(format!("Failed on {:?}: {:?}", &sha256, e));
            }
        },
    };
    response
}

fn aprox_knn_response(ct: Arc<CoverTree>, features: Vec<f32>, k: usize) -> Response<Body> {
    let mut response = Response::new(Body::empty());
    match ct.center_knn_query(&features, k) {
        Ok(trace) => {
            let tr = serde_json::to_string(&trace).unwrap();
            *response.body_mut() = Body::from(tr);
        }
        Err(e) => {
            *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
            *response.body_mut() = Body::from(format!("Failed on trace with: {:?}", e));
        }
    };
    response
}



fn get_k_query(expr: &str) -> usize {
    lazy_static! {
        static ref RE: Regex = Regex::new(r"k=(?P<k>\d+)").unwrap();
    }
    match RE.captures(expr) {
        Some(caps) => caps["k"].parse::<usize>().unwrap(),
        None => 10,
    }
}

impl Service for CoverTreeService {
    type ReqBody = Body;
    type ResBody = Body;
    type Error = hyper::Error;
    type Future = BoxFut;

    /// This is our service handler. It receives a Request, routes on its
    /// path, and returns a Future of a Response.
    fn call(&mut self, req: Request<Body>) -> BoxFut {
        let ct = Arc::clone(&self.ct);
        match (req.method(), req.uri().path()) {
            // Serve some instructions at /
            (&Method::GET, "/parameters/") => {
                let (sb, co, re) = ct.parameters();
                Box::new(future::ok(Response::new(Body::from(format!(
                    "{{\"scale_base\":{},\"cutoff\":{},\"resolution\":{}}}",
                    sb, co, re
                )))))
            }

            (&Method::GET, "/metaschema/") => {
                Box::new(future::ok(Response::new(Body::from(ct.metadata_schema()))))
            }

            (&Method::POST, "/trace/named") => {
                Box::new(req.into_body().concat2().and_then(move |v| {
                    let sha256 = String::from_utf8(v.to_vec()).unwrap();
                    Ok(named_trace_response(ct, sha256))
                }))
            }

            (&Method::POST, "/trace/features") => {
                let file_type = FileType::parse(&req);
                Box::new(req.into_body().concat2().and_then(move |packed| {
                    match file_type.read_vec(packed.to_vec()) {
                        Ok(features) => Ok(trace_response(ct, features)),
                        Err(e) => {
                            let mut response = Response::new(Body::empty());
                            *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                            *response.body_mut() = Body::from(format!("Failed: {:?}", e));
                            return Ok(response);
                        }
                    }
                }))
            }

            (&Method::POST, "/aprox_knn/named") => {
                let k: usize = match req.uri().query() {
                    Some(k_maybe) => get_k_query(k_maybe),
                    None => 10,
                };
                Box::new(req.into_body().concat2().and_then(move |v| {
                    let sha256 = String::from_utf8(v.to_vec()).unwrap();
                    Ok(named_aprox_knn_response(ct, sha256, k))
                }))
            }

            (&Method::POST, "/aprox_knn/features") => {
                let k: usize = match req.uri().query() {
                    Some(k_maybe) => get_k_query(k_maybe),
                    None => 10,
                };
                let file_type = FileType::parse(&req);
                Box::new(req.into_body().concat2().and_then(move |packed| {
                    match file_type.read_vec(packed.to_vec()) {
                        Ok(features) => Ok(aprox_knn_response(ct, features, k)),
                        Err(e) => {
                            let mut response = Response::new(Body::empty());
                            *response.status_mut() = StatusCode::INTERNAL_SERVER_ERROR;
                            *response.body_mut() = Body::from(format!("Failed: {:?}", e));
                            Ok(response)
                        }
                    }
                }))
            }

            // The 404 Not Found route...
            _ => {
                let mut response = Response::new(Body::empty());
                *response.status_mut() = StatusCode::NOT_FOUND;
                Box::new(future::ok(response))
            }
        }
    }
}

impl IntoFuture for CoverTreeService {
    type Future = future::FutureResult<Self::Item, Self::Error>;
    type Item = Self;
    type Error = hyper::Error;

    fn into_future(self) -> Self::Future {
        future::ok(self)
    }
}

