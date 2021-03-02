use std::sync::{RwLock,Arc};

use goko::CoverTreeWriter;
use pointcloud::*;

use tower::Service;

use std::convert::Infallible;

use futures::future;
use core::task::Context;
use std::task::Poll;

use std::marker::PhantomData;
use std::ops::Deref;

use super::GokoHttp;
use crate::parsers::{PointParser, PointBuffer};
use crate::core::*;

pub struct MakeGokoHttp<D: PointCloud, P> {
    writer: Arc<CoreWriter<D>>,
    parser: PhantomData<P>,
}

impl<D, P> MakeGokoHttp<D, P>
where
    D: PointCloud,
    P: PointParser,
    P::Point: Deref<Target = D::Point> + Send + Sync,
{
    pub fn new(writer: Arc<CoreWriter<D>>) -> MakeGokoHttp<D, P> {
        MakeGokoHttp { 
            writer,
            parser: PhantomData,
        }
    }
}

impl<D, T, P> Service<T> for MakeGokoHttp<D, P>
where
    D: PointCloud,
    P: PointParser,
    P::Point: Deref<Target = D::Point> + Send + Sync + 'static,
{
    type Response = GokoHttp<D, P>;
    type Error = Infallible;
    type Future = futures::future::Ready<Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, _: &mut Context) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _: T) -> Self::Future {
        let reader = self.writer.reader();
        let parser = PointBuffer::<P>::new();
        future::ready(Ok(GokoHttp::new(reader, parser)))
    }
}