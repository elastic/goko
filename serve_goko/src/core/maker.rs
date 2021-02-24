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

use super::GokoCore;
use crate::parsers::PointParser;


pub struct MakeGokoCore<D: PointCloud, P> {
    writer: Arc<RwLock<CoverTreeWriter<D>>>,
    parser: PhantomData<P>,
}

impl<D, P> MakeGokoCore<D, P>
where
    D: PointCloud,
    P: PointParser,
    P::Point: Deref<Target = D::Point> + Send + Sync,
{
    pub fn new(writer: Arc<RwLock<CoverTreeWriter<D>>>) -> MakeGokoCore<D, P> {
        MakeGokoCore { 
            writer,
            parser: PhantomData,
        }
    }
}

impl<D, T, P> Service<T> for MakeGokoCore<D, P>
where
    D: PointCloud,
    P: PointParser,
    P::Point: Deref<Target = D::Point> + Send + Sync + 'static,
{
    type Response = GokoCore<D, P>;
    type Error = Infallible;
    type Future = futures::future::Ready<Result<Self::Response, Self::Error>>;

    fn poll_ready(&mut self, _: &mut Context) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _: T) -> Self::Future {
        let reader = self.writer.read().unwrap().reader();
        let parser = P::new();
        future::ready(Ok(GokoCore::new(reader, parser)))
    }
}