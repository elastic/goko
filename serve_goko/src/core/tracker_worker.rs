use pointcloud::PointCloud;
use goko::{CoverTreeReader,CoverTreeWriter};
use std::sync::{Arc,RwLock};
use crate::{GokoRequest, GokoResponse};
use std::ops::Deref;
use goko::errors::GokoError;
use std::collections::HashMap;

use goko::plugins::discrete::tracker::BayesCovertree;

pub(crate) type TrackerResponseSender = oneshot::Sender<Result<Response<Body>, GokoClientError>>;
pub(crate) type TrackerResponseReciever = oneshot::Receiver<Result<Response<Body>, GokoClientError>>;
pub(crate) type TrackerRequestSender = mpsc::UnboundedSender<Message>;
pub(crate) type TrackerRequestReciever = mpsc::UnboundedReceiver<Message>;

pub struct TrackerWorker<D: PointCloud> {
    in_flight: Arc<atomic::AtomicU32>,
    request_snd: TrackerResponseSender,
    pointcloud: PhantomData<D>,
    global_error: Arc<Mutex<Option<Box<dyn std::error::Error + Send>>>>,
}

impl<D> TrackerWorker<D>
where
    D: PointCloud,
{
    pub(crate) fn new<P>(mut tracker: BayesCovertree<D>) -> TrackerWorker<D> 
    where P: Deref<Target = D::Point> + Send + Sync + 'static {
        let (request_snd, mut request_rcv): (TrackerRequestSender<P>, TrackerRequestReciever<P>) =
            mpsc::unbounded_channel();
        tokio::spawn(async move {
            while let Some(mut msg) = request_rcv.recv().await {
                    let response = match goko_request {
                        Ok(r) => reader.process(r).await.map_err(|e| e.into()),
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
        TrackerWorker {
            in_flight,
            request_snd,
            pointcloud: PhantomData,
            global_error,
        }
    }

    pub(crate) fn message(&self, request: Request<Body>) -> ResponseFuture {
        let flight_counter = Arc::clone(&self.in_flight);
        self.in_flight.fetch_add(1, atomic::Ordering::SeqCst);
        let (reply, response): (TrackerResponseSender, TrackerResponseReciever) = oneshot::channel();
        
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