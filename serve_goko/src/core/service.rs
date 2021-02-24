use tokio::sync::{mpsc, oneshot};
use crate::GokoResponse;

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

use super::message::*;
use super::GokoCoreError;
use crate::PointParser;
use crate::api::{parse_http,into_http};

pub struct GokoCore<D: PointCloud, P: PointParser> {
    in_flight: Arc<atomic::AtomicU32>,
    request_snd: CoreRequestSender,
    pointcloud: PhantomData<D>,
    parser: PhantomData<P>,
    global_error: Arc<Mutex<Option<Box<dyn std::error::Error + Send>>>>,
}



impl<D, P> GokoCore<D, P>
where
    D: PointCloud,
    P: PointParser,
    P::Point: Deref<Target = D::Point> + Send + Sync + 'static,
{
    pub fn new(reader: CoverTreeReader<D>, parser: P) -> GokoCore<D, P> {
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
                    msg.error(GokoCoreError::DoubleRead)
                }
            }
        });
        let global_error = Arc::new(Mutex::new(None));
        let in_flight = Arc::new(atomic::AtomicU32::new(0));
        GokoCore {
            in_flight,
            request_snd,
            pointcloud: PhantomData,
            parser: PhantomData,
            global_error,
        }
    }

    pub(crate) fn message(&self, request: P::Future) -> (Message<D::Name,P::Point,P::Future>, CoreResponseReciever<D::Name>) {
        let (reply, response): (CoreResponseSender<D::Name>, CoreResponseReciever<D::Name>) = oneshot::channel();
        
        let msg = Message {
            request: Some(request),
            reply: Some(reply),
            global_error: Arc::clone(&self.global_error),
        };
        (msg, response)
    }
}

impl<D,P> Service<Request<Body>> for GokoCore<D, P> 
where
    D: PointCloud,
    P: PointParser,
    P::Point: Deref<Target = D::Point> + Send + Sync + 'static,
{
    type Response = Response<Body>;
    type Error = GokoCoreError;
    type Future = ResponseFuture<D::Name>;

    fn poll_ready(&mut self, _: &mut Context) -> Poll<Result<(), Self::Error>> {
        if self.request_snd.is_closed() {
            Poll::Ready(Err(GokoCoreError::ClientDropped))
        } else {
            Poll::Ready(Ok(()))
        }
    }

    fn call(&mut self, req: P::Future) -> Self::Future {
        let (message,response) = self.message(req);
        let flight_counter = Arc::clone(&self.in_flight);
        self.in_flight.fetch_add(1, atomic::Ordering::SeqCst);

        let error = self.request_snd.send(message).err().map(|_e| GokoCoreError::FailedSend); 
        ResponseFuture {
            response,
            flight_counter,
            error,
        }
    }
}

impl<D: PointCloud,P: Parser> Load for GokoCore<D, P> {
    type Metric = u32;
    fn load(&self) -> Self::Metric {
        self.in_flight.load(atomic::Ordering::SeqCst)
    }
}