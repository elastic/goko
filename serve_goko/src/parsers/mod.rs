use http::Request;
use hyper::Body;

use core::task::Context;
use std::future::Future;
use std::pin::Pin;
use std::task::Poll;

use crate::errors::*;
use std::fmt::Debug;
use std::marker::PhantomData;

use hyper::body::HttpBody;
use pin_project::pin_project;
use serde::Serialize;

mod msgpack_dense;
pub use msgpack_dense::MsgPackDense;

pub trait PointParser: Send + 'static {
    type Point: Serialize + Send + Sync + Debug + 'static;
    fn parse(
        body_buffer: &[u8],
        scratch_buffer: &mut Vec<u8>,
        request: &Request<Body>,
    ) -> Result<Self::Point, GokoClientError>;
}

#[pin_project]
pub(crate) struct PointBuffer<P: PointParser> {
    body_buffer: Vec<u8>,
    point_buffer: Vec<u8>,
    request: Request<Body>,
    parser: PhantomData<P>,
}

impl<P: PointParser> PointBuffer<P> {
    pub(crate) fn new() -> Self {
        PointBuffer {
            body_buffer: Vec::with_capacity(8 * 1024),
            point_buffer: Vec::with_capacity(8 * 1024),
            request: Request::default(),
            parser: PhantomData,
        }
    }
    pub(crate) fn switch(&mut self, req: Request<Body>) {
        self.request = req;
        self.body_buffer.clear();
        self.point_buffer.clear();
    }

    pub(crate) fn poll_point(
        self: Pin<&mut Self>,
        cx: &mut Context,
    ) -> Poll<Result<P::Point, GokoClientError>> {
        let this = self.project();
        let mut body = this.request.body_mut();
        loop {
            let new_bytes = match Pin::new(&mut body).poll_data(cx) {
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
                        *this.request = Request::default();
                        return Poll::Ready(Err(e.into()));
                    }
                }
            } else {
                match Pin::new(&mut body).poll_trailers(cx) {
                    Poll::Ready(_) => (),
                    Poll::Pending => return Poll::Pending,
                }
            }

            if body.is_end_stream() {
                let point_res = P::parse(this.body_buffer, this.point_buffer, this.request);
                this.body_buffer.clear();
                this.point_buffer.clear();
                *this.request = Request::default();
                return Poll::Ready(point_res);
            }
        }
    }

    pub(crate) fn point(&mut self, req: Request<Body>) -> PointFuture<'_, P>
    where
        Self: Unpin + Sized,
    {
        self.switch(req);
        PointFuture { req: self }
    }
}

#[pin_project]
/// Future that resolves to the next data chunk from `Body`
pub(crate) struct PointFuture<'a, P: PointParser> {
    req: &'a mut PointBuffer<P>,
}

impl<'a, P: PointParser> Future for PointFuture<'a, P> {
    type Output = Result<P::Point, GokoClientError>;

    fn poll(mut self: Pin<&mut Self>, ctx: &mut Context<'_>) -> Poll<Self::Output> {
        Pin::new(&mut *self.req).poll_point(ctx)
    }
}
