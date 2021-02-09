use goko::{CoverTreeReader, CoverTreeWriter};
use http::{Request, Response};
use hyper::Body;
use pointcloud::*;

use tower::Service;
use tower::load::Load;

use tokio::sync::{mpsc,oneshot};
use std::convert::Infallible;

use std::task::Poll;
use core::task::Context;
use std::future::Future;
use std::pin::Pin;

use std::sync::{Arc, atomic};

use std::marker::PhantomData;
use std::ops::Deref;

use crate::parser::*;
use crate::errors::*;
use crate::{process, parse_request};

type ResponseSender = oneshot::Sender<Result<Response<Body>,hyper::Error>>;
type ResponseReciever = oneshot::Receiver<Result<Response<Body>,hyper::Error>>;

type RequestSender = mpsc::UnboundedSender<(Request<Body>,ResponseSender)>;
type RequestReciever = mpsc::UnboundedReceiver<(Request<Body>,ResponseSender)>;

pub struct GokoClient<D> {
    in_flight: Arc<atomic::AtomicU32>,
    request_snd: RequestSender,
    pointcloud: PhantomData<D>,
}

impl<D: PointCloud> GokoClient<D>
    where 
    D: PointCloud,
     {
    pub fn new<'a,P>(parser: P, reader: CoverTreeReader<D>) -> GokoClient<D>
    where
    P: ParserService,
    P::Point: Deref<Target = D::Point> + Send + Sync,
     {
        let (request_snd, mut request_rcv): (RequestSender, RequestReciever) =  mpsc::unbounded_channel();
        tokio::spawn(async move {
            while let Some((request,response_tx)) = request_rcv.recv().await {
                match parse_request(&parser,request).await {
                    Ok(goko_request) => response_tx.send(Ok(process(goko_request, &reader))),
                    Err(e) => response_tx.send(Err(e)),
                }.unwrap();
            };
        });
        let in_flight = Arc::new(atomic::AtomicU32::new(0));
        GokoClient {
            in_flight,
            request_snd,
            pointcloud: PhantomData,
        }
    }
}


impl<D> Service<Request<Body>> for GokoClient<D> {
    type Response = Response<Body>;
    type Error = GokoClientError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _: &mut Context) -> Poll<Result<(), Self::Error>> {
        if self.request_snd.is_closed() {
            Poll::Ready(Err(GokoClientError::ClientDropped))
        } else {
            Poll::Ready(Ok(()))
        }
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        let in_flight_req = Arc::clone(&self.in_flight);
        self.in_flight.fetch_add(1, atomic::Ordering::SeqCst);
        let (tx,rx): (ResponseSender,ResponseReciever) = oneshot::channel();
        match self.request_snd.send((req,tx)) {
            Ok(_) => {
                Box::pin(async move { 
                    let res = rx.await.unwrap().map_err(|e| e.into());
                    in_flight_req.fetch_sub(1, atomic::Ordering::SeqCst);
                    res
                 })
            },
            Err(e) => {
                Box::pin(async {Err(e.into())})
            }
        }
        
    }
}

impl<D> Load for GokoClient<D> {
    type Metric = u32;
    fn load(&self) -> Self::Metric {
        self.in_flight.load(atomic::Ordering::SeqCst)
    }
}

pub struct MakeGokoService<D: PointCloud, P> {
    writer: CoverTreeWriter<D>,
    parser: P,
}

impl<'a, D,P> MakeGokoService<D, P> 
    where 
    D: PointCloud,
    P: ParserService,
    P::Point: Deref<Target = D::Point> + Send + Sync,
{
    pub fn new(parser: P, writer: CoverTreeWriter<D>) -> MakeGokoService<D, P>
    {
        MakeGokoService {
            parser,
            writer,
        }
    }
}

impl<'a, P, D, T> Service<T> for MakeGokoService<D, P>
    where 
        D: PointCloud,
        P: ParserService + Clone,
        P::Point: Deref<Target = D::Point> + Send + Sync, 
    {
    type Response = GokoClient<D>;
    type Error = Infallible;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _: &mut Context) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _: T) -> Self::Future {
        let reader = self.writer.reader();
        let parser = self.parser.clone();
        Box::pin(async {Ok(GokoClient::new(parser, reader))})
    }
}