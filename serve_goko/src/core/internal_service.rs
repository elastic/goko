use tokio::sync::{mpsc, oneshot};
use pin_project::pin_project;
use goko::errors::GokoError;
use core::task::Context;
use std::future::Future;
use std::pin::Pin;
use std::task::Poll;

use crate::errors::*;

use std::sync::{atomic, Arc, Mutex};

pub(crate) type CoreRequestSender<T,S> = mpsc::UnboundedSender<Message<T,S>>;
pub(crate) type CoreRequestReciever<T,S> = mpsc::UnboundedReceiver<Message<T,S>>;
pub(crate) type CoreResponseSender<T> = oneshot::Sender<Result<T,InternalServiceError>>;
pub(crate) type CoreResponseReciever<T> = oneshot::Receiver<Result<T,InternalServiceError>>;


#[pin_project]
pub(crate) struct Message<T: Send,S: Send> {
    pub(crate) request: Option<T>,
    pub(crate) reply: Option<CoreResponseSender<S>>,
    pub(crate) global_error: Arc<Mutex<Option<Box<dyn std::error::Error + Send>>>>,
}

impl<T: Send, S: Send> Message<T, S> {
    pub(crate) fn request(&mut self) -> Option<T> {
        self.request.take()
    }

    pub(crate) fn respond(&mut self, response: Result<S,GokoError>) {
        match self.reply.take() {
            Some(reply) => {
                match reply.send(response.map_err(|e| InternalServiceError::from(e))) {
                    Ok(_) => (),
                    Err(_) => {
                        *self.global_error.lock().unwrap() = Some(Box::new(InternalServiceError::FailedRespSend));
                    }
                }
            }
            None => *self.global_error.lock().unwrap() = Some(Box::new(InternalServiceError::DoubleRead)),
        }
    }
    pub(crate) fn error(&mut self, error: impl std::error::Error + Send + 'static) {
        *self.global_error.lock().unwrap() = Some(Box::new(error));
    }
}

#[pin_project]
pub struct ResponseFuture<T> {
    #[pin]
    pub(crate) response: oneshot::Receiver<Result<T,InternalServiceError>>,
    pub(crate) flight_counter: Arc<atomic::AtomicU32>,
    pub(crate) error: Option<InternalServiceError>,
}

impl<T> Future for ResponseFuture<T> {
    type Output = Result<T, InternalServiceError>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        if let Some(err) = this.error.take() {
            return core::task::Poll::Ready(Err(err));
        }
        else {
            let res = this.response.poll(cx).map(|r| {
                match r {
                    Ok(r) => r.map_err(|e| e.into()),
                    Err(e) => Err(e.into())
                }
            });
            this.flight_counter.fetch_sub(1, atomic::Ordering::SeqCst);
            res
        }
    }
}

pub trait InternalService<T, S>: Send {
    fn process(&mut self, request: T) -> Result<S, GokoError>;
}

#[derive(Clone)]
pub(crate) struct InternalServiceOperator<T: Send, S: Send> {
    in_flight: Arc<atomic::AtomicU32>,
    request_snd: CoreRequestSender<T,S>,
    global_error: Arc<Mutex<Option<Box<dyn std::error::Error + Send>>>>,
}

impl<T: Send + 'static, S: Send + 'static> InternalServiceOperator<T, S> {

    pub(crate) fn new<P: InternalService<T,S> + 'static>(mut server: P) -> InternalServiceOperator<T, S> {
        let (request_snd, mut request_rcv): (CoreRequestSender<T,S>, CoreRequestReciever<T,S>) =
            mpsc::unbounded_channel();
        tokio::spawn(async move {
            while let Some(mut msg) = request_rcv.recv().await {
                if let Some(request) = msg.request() {
                    let response = server.process(request);
                    msg.respond(response);
                } else {
                    msg.error(InternalServiceError::DoubleRead)
                }
            }
        });
        let global_error = Arc::new(Mutex::new(None));
        let in_flight = Arc::new(atomic::AtomicU32::new(0));
        InternalServiceOperator {
            in_flight,
            request_snd,
            global_error,
        }
    }


    pub(crate) fn message(&self, request: T) -> ResponseFuture<S> {
        let flight_counter = Arc::clone(&self.in_flight);
        self.in_flight.fetch_add(1, atomic::Ordering::SeqCst);
        let (reply, response): (CoreResponseSender<S>, CoreResponseReciever<S>) = oneshot::channel();
        
        let msg = Message {
            request: Some(request),
            reply: Some(reply),
            global_error: Arc::clone(&self.global_error),
        };

        let error = self.request_snd.send(msg).err().map(|_e| InternalServiceError::FailedSend); 
        ResponseFuture {
            response,
            flight_counter,
            error,
        }
    }
}
