use tokio::sync::{mpsc, oneshot};

use http::{Uri, Method, Request, Response};
use hyper::Body;

use crate::{GokoRequest,GokoResponse};

use goko::CoverTreeReader;
use pointcloud::*;

use tower::load::Load;
use tower::Service;

use core::task::Context;
use std::task::Poll;

use crate::api::Process;

use std::sync::{atomic, Arc, Mutex};

use std::marker::PhantomData;
use std::ops::Deref;
use regex::Regex;
use lazy_static::lazy_static;
use super::message::*;
use super::GokoHttpError;
use crate::PointParser;
use crate::parsers::PointBuffer;
use crate::errors::*;
use crate::api::*;
use crate::core::*;


pub struct GokoHttp<D: PointCloud, P: PointParser> {
    in_flight: Arc<atomic::AtomicU32>,
    request_snd: CoreRequestSender,
    pointcloud: PhantomData<D>,
    parser: PhantomData<P>,
    global_error: Arc<Mutex<Option<Box<dyn std::error::Error + Send>>>>,
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

pub(crate) async fn parse_http<P: PointParser>(request: Request<Body>, parser: &mut PointBuffer<P>) -> Result<GokoRequest<P::Point>, GokoClientError> {
    match (request.method(), request.uri().path()) {
        // Serve some instructions at /
        (&Method::GET, "/") => Ok(GokoRequest::Parameters(ParametersRequest)),
        (&Method::GET, "/knn") => {
            let k = parse_knn_query(request.uri());
            let point = parser.point(request).await?;
            Ok(GokoRequest::Knn(KnnRequest { point, k }))
        }
        (&Method::GET, "/routing_knn") => {
            let k = parse_knn_query(request.uri());
            let point = parser.point(request).await?;
            Ok(GokoRequest::RoutingKnn(RoutingKnnRequest { point, k }))

        }
        (&Method::GET, "/path") => {
            let point = parser.point(request).await?;
            Ok(GokoRequest::Path(PathRequest { point }))

        }
        // The 404 Not Found route...
        _ => Ok(GokoRequest::Unknown(String::new(), 404)),
    }
}

pub(crate) fn into_http(response: GokoResponse) -> Result<Response<Body>, GokoClientError> {
    let mut builder = http::response::Builder::new();
    let json_str = match response {
        GokoResponse::Parameters(p) => serde_json::to_string(&p).unwrap(),
        GokoResponse::Knn(p) => serde_json::to_string(&p).unwrap(),
        GokoResponse::RoutingKnn(p) => serde_json::to_string(&p).unwrap(),
        GokoResponse::Path(p) => serde_json::to_string(&p).unwrap(),
        GokoResponse::Unknown(response_string, status) => {
            builder = builder.status(status);
            response_string
        }
    };
    Ok(builder.body(Body::from(json_str)).unwrap())
}

impl<D, P> GokoHttp<D, P>
where
    D: PointCloud,
    P: PointParser,
    P::Point: Deref<Target = D::Point> + Send + Sync + 'static,
{
    pub(crate) fn new(reader: CoreReader<D>, mut parser: PointBuffer<P>) -> GokoHttp<D, P> {
        let (request_snd, mut request_rcv): (CoreRequestSender, CoreRequestReciever) =
            mpsc::unbounded_channel();
        tokio::spawn(async move {
            while let Some(mut msg) = request_rcv.recv().await {
                if let Some(hyper_request) = msg.request() {
                    let goko_request = parse_http(hyper_request, &mut parser).await;
                    let response = match goko_request {
                        Ok(r) => r.process(&reader).map_err(|e| e.into()),
                        Err(e) => Err(e),
                    };
                    match response {
                        Ok(resp) => msg.respond(into_http(resp)),
                        Err(e) => msg.respond(Err(e)),
                    };
                } else {
                    msg.error(GokoHttpError::DoubleRead)
                }
            }
        });
        let global_error = Arc::new(Mutex::new(None));
        let in_flight = Arc::new(atomic::AtomicU32::new(0));
        GokoHttp {
            in_flight,
            request_snd,
            pointcloud: PhantomData,
            parser: PhantomData,
            global_error,
        }
    }

    pub(crate) fn message(&self, request: Request<Body>) -> ResponseFuture {
        let flight_counter = Arc::clone(&self.in_flight);
        self.in_flight.fetch_add(1, atomic::Ordering::SeqCst);
        let (reply, response): (CoreResponseSender, CoreResponseReciever) = oneshot::channel();
        
        let msg = Message {
            request: Some(request),
            reply: Some(reply),
            global_error: Arc::clone(&self.global_error),
        };

        let error = self.request_snd.send(msg).err().map(|_e| GokoHttpError::FailedSend); 
        ResponseFuture {
            response,
            flight_counter,
            error,
        }
    }
}

impl<D,P> Service<Request<Body>> for GokoHttp<D, P> 
where
    D: PointCloud,
    P: PointParser,
    P::Point: Deref<Target = D::Point> + Send + Sync + 'static,
{
    type Response = Response<Body>;
    type Error = GokoHttpError;
    type Future = ResponseFuture;

    fn poll_ready(&mut self, _: &mut Context) -> Poll<Result<(), Self::Error>> {
        if self.request_snd.is_closed() {
            Poll::Ready(Err(GokoHttpError::ClientDropped))
        } else {
            Poll::Ready(Ok(()))
        }
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        self.message(req)
    }
}

impl<D: PointCloud,P: PointParser> Load for GokoHttp<D, P> {
    type Metric = u32;
    fn load(&self) -> Self::Metric {
        self.in_flight.load(atomic::Ordering::SeqCst)
    }
}