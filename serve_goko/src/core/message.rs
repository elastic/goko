use tokio::sync::{mpsc, oneshot};
use crate::{GokoRequest,GokoResponse};
use pin_project::pin_project;

use core::task::Context;
use std::future::Future;
use std::pin::Pin;
use std::task::Poll;

use std::sync::{atomic, Arc, Mutex};

use crate::errors::GokoClientError;
use super::*;

pub(crate) type CoreResponseSender = oneshot::Sender<Result<Response<Body>, GokoClientError>>;
pub(crate) type CoreResponseReciever = oneshot::Receiver<Result<Response<Body>, GokoClientError>>;
pub(crate) type CoreRequestSender = mpsc::UnboundedSender<Message>;
pub(crate) type CoreRequestReciever = mpsc::UnboundedReceiver<Message>;

#[pin_project]
pub(crate) struct Message {
    pub(crate) request: Option<Result<Request<Body>,GokoClientError>>,
    pub(crate) reply: Option<CoreResponseSender>,
    pub(crate) global_error: Arc<Mutex<Option<Box<dyn std::error::Error + Send>>>>,
}

impl Message {
    pub(crate) fn request(&mut self) -> Option<Result<Request<Body>,GokoClientError>> {
        self.request.take()
    }

    pub(crate) fn respond(&mut self, response: Result<Response<Body>,GokoClientError>) {
        match self.reply.take() {
            Some(reply) => {
                match reply.send(response) {
                    Ok(_) => (),
                    Err(_) => {
                        *self.global_error.lock().unwrap() = Some(Box::new(GokoCoreError::FailedRespSend));
                    }
                }
            }
            None => *self.global_error.lock().unwrap() = Some(Box::new(GokoCoreError::DoubleRead)),
        }
    }
    pub(crate) fn error(&mut self, error: impl std::error::Error + Send + 'static) {
        *self.global_error.lock().unwrap() = Some(Box::new(error));
    }
} 

#[pin_project]
pub struct ResponseFuture {
    #[pin]
    pub(crate) response: CoreResponseReciever,
    pub(crate) flight_counter: Arc<atomic::AtomicU32>,
    pub(crate) error: Option<GokoCoreError>,
}

impl<N: Send> Future for ResponseFuture {
    type Output = Result<Response<Body>, GokoCoreError>;
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
