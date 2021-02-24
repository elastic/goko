use warp::Filter;
use serde::Serialize;
use crate::GokoRequest;
use crate::errors::GokoClientError;
use crate::api::knn::KnnRequest;
use crate::api::parameters::ParametersRequest;
use bytes::Bytes;
use std::fmt::Debug;
use std::marker::PhantomData;
use tower::{Layer, Service, MakeService};
use warp::filters::BoxedFilter;

impl<P: Debug + Sized + Send + Sync + 'static> warp::reject::Reject for GokoClientError<P> {}

pub trait Parser: Send + Sync + 'static {
    type Point: Serialize + Send + Sync + Debug + 'static;
    fn parse(bytes: Bytes) -> Result<Self::Point, GokoClientError<Self::Point>>;
}

pub fn rest_filters<P: Parser + Clone>() -> BoxedFilter<(GokoRequest<P::Point>,)> {
    parameters::<P>().or(knn::<P>()).boxed()
}

/// GET /todos?offset=3&limit=5
pub fn parameters<P: Parser + Clone>() -> impl Filter<Extract = (GokoRequest<P::Point>,), Error = warp::Rejection> + Clone {
    warp::path::end().map(|| GokoRequest::Parameters(ParametersRequest))
}

/// GET /todos?offset=3&limit=5
pub fn knn<P: Parser + Clone>() -> impl Filter<Extract = (GokoRequest<P::Point>,), Error = warp::Rejection> + Clone {
    warp::path!("knn" / usize).and(warp::body::bytes()).and_then(|k: usize, body: Bytes| async move {
        
        match P::parse(body) {
            Ok(point) => {
                Ok(GokoRequest::Knn(KnnRequest {
                    k,
                    point,
                }))
            }
            Err(e) => {
                Err(warp::reject::custom(e))
            }
        }
    })
}
/*
pub struct RestLayer<P> {
    target: PhantomData<P>,
}

impl<P,S> Layer<S> for RestLayer<P> {
    type Service = RestService<P,S>;

    fn layer(&self, service: S) -> Self::Service {
        RestService {
            filter: warp::service(rest_filters::<P>()),
            service,
        }
    }
}

pub struct RestService<F,S> {
    filter: F,
    service: S,
}

impl<S, Request> Service<Request> for RestService<S>
where
    S: Service<Request>,
    Request: fmt::Debug,
{
    type Response = S::Response;
    type Error = S::Error;
    type Future = S::Future;

    fn poll_ready(&mut self, cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        self.service.poll_ready(cx)
    }

    fn call(&mut self, request: Request) -> Self::Future {
        // Insert log statement here or other functionality
        println!("request = {:?}, target = {:?}", request, self.target);
        self.service.call(request)
    }
}
*/
/*
/// The 4 TODOs filters combined.
pub fn todos(
    db: Db,
) -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
    todos_list(db.clone())
        .or(todos_create(db.clone()))
        .or(todos_update(db.clone()))
        .or(todos_delete(db))
}
/// POST /todos with JSON body
pub fn todos_create(
    db: Db,
) -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
    warp::path!("todos")
        .and(warp::post())
        .and(json_body())
        .and(with_db(db))
        .and_then(handlers::create_todo)
}

/// PUT /todos/:id with JSON body
pub fn todos_update(
    db: Db,
) -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
    warp::path!("todos" / u64)
        .and(warp::put())
        .and(json_body())
        .and(with_db(db))
        .and_then(handlers::update_todo)
}

/// DELETE /todos/:id
pub fn todos_delete(
    db: Db,
) -> impl Filter<Extract = impl warp::Reply, Error = warp::Rejection> + Clone {
    // We'll make one of our endpoints admin-only to show how authentication filters are used
    let admin_only = warp::header::exact("authorization", "Bearer admin");

    warp::path!("todos" / u64)
        // It is important to put the auth check _after_ the path filters.
        // If we put the auth check before, the request `PUT /todos/invalid-string`
        // would try this filter and reject because the authorization header doesn't match,
        // rather because the param is wrong for that other path.
        .and(admin_only)
        .and(warp::delete())
        .and(with_db(db))
        .and_then(handlers::delete_todo)
}

fn with_db(db: Db) -> impl Filter<Extract = (Db,), Error = std::convert::Infallible> + Clone {
    warp::any().map(move || db.clone())
}

fn json_body() -> impl Filter<Extract = (Todo,), Error = warp::Rejection> + Clone {
    // When accepting a body, we want a JSON body
    // (and to reject huge payloads)...
    warp::body::content_length_limit(1024 * 16).and(warp::body::json())
}
*/