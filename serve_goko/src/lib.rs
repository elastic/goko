#![deny(warnings)]

use warp::Filter;
use goko::{CoverTreeReader, CoverTreeWriter};
use pointcloud::*;

use tower::Layer;
use hyper::{Body,Response};
use hyper::service::Service;

use std::task::Poll;
use core::task::Context;
use std::future::Future;
use std::pin::Pin;

pub enum GokoQuery {
    Parameters,
    Knn(Vec<f32>,usize),
}

pub fn tree_router() -> impl Filter<Extract = (GokoQuery,), Error = warp::Rejection> + Copy {
    let tree_parameters = warp::path("parameters").map(|| GokoQuery::Parameters);

    tree_parameters
}

pub struct GokoService<D: PointCloud> {
    pub reader: CoverTreeReader<D>,
}

impl<D: PointCloud> Service<GokoQuery> for GokoService<D> {
    type Response = Response<Body>;
    type Error = hyper::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>>>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: GokoQuery) -> Self::Future {
        let response = match req {
            GokoQuery::Parameters => {
                let params = self.reader.parameters();
                Response::new(Body::from(format!(
                    "{{\"scale_base\":{},\"leaf_cutoff\":{},\"min_res_index\":{}}}",
                    params.scale_base, params.leaf_cutoff, params.min_res_index
                )))
            }
            GokoQuery::Knn(point,k) => {
                Response::new(Body::from("KNN GOES HERE"))
            }
        };
         
        // create a response in a future.
        let fut = async {
            Ok(response)
        };

        // Return the response as an immediate future
        Box::pin(fut)
    }
}

pub struct GokoLayer<D: PointCloud> {
    pub writer: CoverTreeWriter<D>,
}

impl<T, D: PointCloud> Layer<T> for GokoLayer<D> {
    type Service = GokoService<D>;
    
    fn layer(&self, service: T) -> Self::Service {
        let router = warp::service(tree_router);
        let reader = self.writer.reader();
        GokoService { reader }
    }
}

