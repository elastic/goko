use http::{Method, Request, HeaderValue, Uri};
use hyper::Body;
use tower::Service;
use tower::Layer;

use core::task::Context;
use std::future::Future;
use std::pin::Pin;
use std::task::Poll;

use std::marker::PhantomData;

use hyper::body::to_bytes;
use futures::prelude::*;

use crate::GokoRequest;
use crate::errors::*;
use crate::api::knn::{KnnRequest, RoutingKnnRequest};
use crate::api::parameters::ParametersRequest;
use crate::api::path::PathRequest;

use regex::Regex;
use lazy_static::lazy_static;
use bytes::Bytes;
use std::fmt::Debug;

use serde::Serialize;

pub trait PointParser: Send + Sync + 'static {
    type Point: Serialize + Send + Sync + Debug + 'static;
    type: Future: Future<Output = Result<Self::Point,GokoClientError>>,
    fn parse_body(bytes: Bytes) -> Self::Future;
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

pub struct RestLayer<P> {
    parser: PhantomData<P>,
}

impl<P,S> Layer<S> for RestLayer<P> {
    type Service = RestService<P,S>;

    fn layer(&self, service: S) -> Self::Service {
        RestService {
            service,
            parser: PhantomData,
        }
    }
}

pub struct RestService<P,S> {
    parser: PhantomData<P>,
    service: S,
}

pub struct Buffer {
    bytes: Vec<u8>,
    body: Option<Body>,
}

impl Stream for Buffer {
    type Item = Result<(), hyper::Error>;
    fn poll_next(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Option<Result<(), hyper::Error>>> {
        if let Some(body) = self.body {
            Pin::new(&mut body).poll_next(cx).map_ok(|datum| self.bytes.extend_from_slice(&datum))
        } else {
            Poll::Ready(None)
        }
    }
}

impl<P: Parser, S: Service<GokoRequest<P::Point>> + Send> Service<Request<Body>> for RestService<P, S> {
    type Response = S::Response;
    type Error = GokoClientError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, cx: &mut Context) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, request: Request<Body>) -> Self::Future {
        Box::pin(async { 
            let (parts, body) = request.into_parts();
            let request = match (parts.method, parts.uri.path()) {
                // Serve some instructions at /
                (Method::GET, "/") => GokoRequest::Parameters(ParametersRequest),
                (Method::GET, "/knn") => {
                    let k = parse_knn_query(&parts.uri);
                    let bytes: Bytes = to_bytes(body).await?;
                    let point = P::parse_body(parts.headers.get("Content-Type"), bytes)?;
                    GokoRequest::Knn(KnnRequest { point, k })
                }
                (Method::GET, "/routing_knn") => {
                    let k = parse_knn_query(&parts.uri);
                    let bytes: Bytes = to_bytes(body).await?;
                    let point = P::parse_body(parts.headers.get("Content-Type"), bytes)?;
                    GokoRequest::RoutingKnn(RoutingKnnRequest { point, k })
                }
                (Method::GET, "/path") => {
                    let bytes: Bytes = to_bytes(body).await?;
                    let point = P::parse_body(parts.headers.get("Content-Type"), bytes)?;
                    GokoRequest::Path(PathRequest { point })
                }
                // The 404 Not Found route...
                _ => GokoRequest::Unknown(String::new(), 404),
            };
            self.service.call(request).await
        })
    }
}
