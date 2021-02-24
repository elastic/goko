use http::{Method, Request, HeaderValue, Uri};
use hyper::Body;
use tower::Service;
use tower::Layer;

use core::task::Context;
use std::future::Future;
use std::pin::Pin;
use std::task::Poll;

use crate::GokoRequest;
use crate::errors::*;
use crate::api::knn::{KnnRequest, RoutingKnnRequest};
use crate::api::parameters::ParametersRequest;
use crate::api::path::PathRequest;

use regex::Regex;
use lazy_static::lazy_static;
use std::fmt::Debug;

use serde::Serialize;
use hyper::body::HttpBody;
use pin_project::pin_project;

use super::RequestParser;

pub trait PointParser: Send + 'static {
    type Point: Serialize + Send + Sync + Debug + 'static;
    type Compression: Default + Send + Sync + Debug + 'static;
    fn compression(req: &Request<Body>) -> Self::Compression;
    fn parse(buffer: &[u8], decompressed_buffer: &mut [u8], token: &Self::Compression) -> Result<Self::Point, GokoClientError>;
}

#[pin_project]
pub(crate) struct PointBuffer<P: PointParser> {
    body_buffer: Vec<u8>,
    point_buffer: Vec<u8>,
    body: Body,
    compression: P::Compression,
}

impl<P: PointParser> PointBuffer<P> {
    fn new() -> Self {
        PointBuffer {
            body_buffer: Vec::with_capacity(8*1024),
            point_buffer: Vec::with_capacity(8*1024),
            body: Body::empty(),
            compression: P::Compression::default(),
        }
    }
    fn switch(&mut self, req: Request<Body>) {
        self.compression = P::compression(&req);
        self.body = req.into_body();
        self.body_buffer.clear();
        self.point_buffer.clear();
    }

    fn poll_point(self: Pin<&mut Self>, cx: &mut Context) -> Poll<Result<P::Point, GokoClientError>> {
        let mut this = self.project();
        loop {
            let new_bytes = match Pin::new(&mut this.body).poll_data(cx) {
                Poll::Ready(data) => data,
                Poll::Pending => return Poll::Pending,
            };
            if let Some(new_bytes) = new_bytes {
                match new_bytes {
                    Ok(new_bytes) => {
                        this.body_buffer.extend_from_slice(&new_bytes);
                    }
                    Err(e) => {
                        this.body_buffer.clear();
                        this.point_buffer.clear();
                        *this.body = Body::empty();
                        *this.compression = P::Compression::default();
                        return Poll::Ready(Err(e.into()))
                    },
                }
            } else {
                match Pin::new(&mut this.body).poll_trailers(cx) {
                    Poll::Ready(_) => (),
                    Poll::Pending => return Poll::Pending,
                }
            }

            if this.body.is_end_stream() {
                let point_res = P::parse(this.body_buffer, this.point_buffer, this.compression);
                this.body_buffer.clear();
                this.point_buffer.clear();
                *this.body = Body::empty();
                *this.compression = P::Compression::default();
                return Poll::Ready(point_res)
            }
        }
    }

    pub(crate) fn point(&mut self, req: Request<Body>) -> PointFuture<'_, Self> 
    where 
    Self: Unpin + Sized,
    {
        self.switch(req);
        PointFuture{
            req: self,
        }
    }
}

#[pin_project]
/// Future that resolves to the next data chunk from `Body`
pub(crate) struct PointFuture<'a, T: ?Sized> { 
    req: &'a mut T,
}

impl<'a, T: RequestParser + Unpin + ?Sized> Future for PointFuture<'a, T> {
    type Output = Result<GokoRequest<T::Point>, GokoClientError>;

    fn poll(mut self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut *self.req).poll_request(ctx)
    }
}
