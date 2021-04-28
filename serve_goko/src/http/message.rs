use tokio::sync::{mpsc, oneshot};
use pin_project::pin_project;

use http::{Request, Response};
use hyper::Body;

use core::task::Context;
use std::future::Future;
use std::pin::Pin;
use std::task::Poll;

use std::sync::{atomic, Arc, Mutex};

use crate::errors::{GokoClientError, InternalServiceError};

pub(crate) type HttpResponseSender = oneshot::Sender<Result<Response<Body>, GokoClientError>>;
pub(crate) type HttpResponseReciever = oneshot::Receiver<Result<Response<Body>, GokoClientError>>;
pub(crate) type HttpRequestSender = mpsc::UnboundedSender<HttpMessage>;
pub(crate) type HttpRequestReciever = mpsc::UnboundedReceiver<HttpMessage>;

#[pin_project]
pub(crate) struct HttpMessage {
    pub(crate) request: Option<Request<Body>>,
    pub(crate) reply: Option<HttpResponseSender>,
    pub(crate) global_error: Arc<Mutex<Option<Box<dyn std::error::Error + Send>>>>,
}

impl HttpMessage {
    pub(crate) fn request(&mut self) -> Option<Request<Body>> {
        self.request.take()
    }

    pub(crate) fn respond(&mut self, response: Result<Response<Body>,GokoClientError>) {
        match self.reply.take() {
            Some(reply) => {
                match reply.send(response) {
                    Ok(_) => (),
                    Err(_) => {
                        *self.global_error.lock().unwrap() = Some(Box::new(GokoClientError::Underlying(InternalServiceError::FailedRespSend)));
                    }
                }
            }
            None => *self.global_error.lock().unwrap() = Some(Box::new(GokoClientError::Underlying(InternalServiceError::DoubleRead))),
        }
    }
    pub(crate) fn error(&mut self, error: impl std::error::Error + Send + 'static) {
        *self.global_error.lock().unwrap() = Some(Box::new(error));
    }
} 

#[pin_project]
pub struct ResponseFuture {
    #[pin]
    pub(crate) response: HttpResponseReciever,
    pub(crate) flight_counter: Arc<atomic::AtomicU32>,
    pub(crate) error: Option<GokoClientError>,
}

impl Future for ResponseFuture {
    type Output = Result<Response<Body>, GokoClientError>;
    fn poll(self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Self::Output> {
        let this = self.project();
        if let Some(err) = this.error.take() {
            return core::task::Poll::Ready(Err(err));
        }
        else {
            let res = this.response.poll(cx).map(|r| {
                match r {
                    Ok(r) => r.map_err(|e| GokoClientError::from(e)),
                    Err(e) => Err(GokoClientError::from(e))
                }
            });
            this.flight_counter.fetch_sub(1, atomic::Ordering::SeqCst);
            res
        }
    }
}
