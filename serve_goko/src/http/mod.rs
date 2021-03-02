use tokio::sync::oneshot;
use crate::errors::GokoClientError;
use crate::GokoRequest;
use goko::errors::GokoError;

use hyper::{Request,Body};

use std::error::Error;
use std::fmt;

mod maker;
mod message;
mod service;

pub use service::GokoHttp;
pub use message::ResponseFuture;
pub use maker::MakeGokoHttp;
use std::future::Future;
use serde::Serialize;

pub enum GokoHttpError {
    Other(GokoClientError),
    FailedSend,
    FailedRecv,
    FailedRespSend,
    DoubleRead,
    ClientDropped,
}

impl From<oneshot::error::RecvError> for GokoHttpError {
    fn from(_e: oneshot::error::RecvError) -> GokoHttpError {
        GokoHttpError::FailedRecv
    }
}

impl From<GokoClientError> for GokoHttpError {
    fn from(e: GokoClientError) -> GokoHttpError {
        GokoHttpError::Other(e)
    }
}

impl From<GokoError> for GokoHttpError {
    fn from(e: GokoError) -> GokoHttpError {
        GokoHttpError::Other(GokoClientError::Underlying(e))
    }
}

impl fmt::Display for GokoHttpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use GokoHttpError::*;
        match *self {
            FailedSend => f.pad("Send Failed"),
            Other(ref se) => fmt::Display::fmt(&se, f),
            FailedRecv => f.pad("Recv Failed"),
            FailedRespSend => f.pad("Unable to Respond, client hung up."),
            DoubleRead => f.pad("Attempted to read a message twice"),
            ClientDropped => f.pad("Client Dropped"),
        }
    }
}

impl fmt::Debug for GokoHttpError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use GokoHttpError::*;
        match *self {
            FailedSend => f.pad("SendFailed"),
            Other(ref se) => write!(f, "Other({:?})", se),
            FailedRecv => f.pad("RecvFailed"),
            FailedRespSend => f.pad("FailedRespSend"),
            DoubleRead => f.pad("DoubleRead"),
            ClientDropped => f.pad("ClientDropped"),
        }
    }
}

impl Error for GokoHttpError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        use GokoHttpError::*;
        match *self {
            FailedSend => None,
            Other(ref e) => e.source(),
            FailedRecv => None,
            FailedRespSend => None,
            DoubleRead => None,
            ClientDropped => None,
        }
    }
}
