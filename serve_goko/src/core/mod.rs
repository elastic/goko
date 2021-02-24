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

pub use service::GokoCore;
pub use message::ResponseFuture;
pub use maker::MakeGokoCore;
use std::future::Future;
use serde::Serialize;

pub enum GokoCoreError {
    Other(GokoClientError),
    FailedSend,
    FailedRecv,
    FailedRespSend,
    DoubleRead,
    ClientDropped,
}

impl From<oneshot::error::RecvError> for GokoCoreError {
    fn from(_e: oneshot::error::RecvError) -> GokoCoreError {
        GokoCoreError::FailedRecv
    }
}

impl From<GokoClientError> for GokoCoreError {
    fn from(e: GokoClientError) -> GokoCoreError {
        GokoCoreError::Other(e)
    }
}

impl From<GokoError> for GokoCoreError {
    fn from(e: GokoError) -> GokoCoreError {
        GokoCoreError::Other(GokoClientError::Underlying(e))
    }
}

impl fmt::Display for GokoCoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use GokoCoreError::*;
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

impl fmt::Debug for GokoCoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use GokoCoreError::*;
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

impl Error for GokoCoreError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        use GokoCoreError::*;
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
