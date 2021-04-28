use std::error::Error;
use std::fmt;
use goko::errors::GokoError;
use tokio::sync::oneshot;

pub enum InternalServiceError {
    Other(GokoError),
    FailedSend,
    FailedRecv,
    FailedRespSend,
    DoubleRead,
    ClientDropped,
}

impl From<oneshot::error::RecvError> for InternalServiceError {
    fn from(_e: oneshot::error::RecvError) -> InternalServiceError {
        InternalServiceError::FailedRecv
    }
}

impl From<GokoError> for InternalServiceError {
    fn from(e: GokoError) -> InternalServiceError {
        InternalServiceError::Other(e)
    }
}

impl fmt::Display for InternalServiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use InternalServiceError::*;
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

impl fmt::Debug for InternalServiceError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        use InternalServiceError::*;
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

impl Error for InternalServiceError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        use InternalServiceError::*;
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

//use serde::{Deserialize, Serialize};
//
pub enum GokoClientError {
    Underlying(InternalServiceError),
    Http(hyper::Error),
    Parse(Box<dyn std::error::Error + Send + Sync>),
    MissingBody,
}

impl GokoClientError {
    pub fn parse(err: Box<dyn std::error::Error + Send + Sync>) -> Self {
        GokoClientError::Parse(err)
    }
}

impl From<GokoError> for GokoClientError {
    fn from(e: GokoError) -> GokoClientError {
        GokoClientError::Underlying(InternalServiceError::Other(e))
    }
}

impl From<oneshot::error::RecvError> for GokoClientError {
    fn from(_e: oneshot::error::RecvError) -> GokoClientError {
        GokoClientError::Underlying(InternalServiceError::FailedRecv)
    }
}

impl From<InternalServiceError> for GokoClientError {
    fn from(e: InternalServiceError) -> GokoClientError {
        GokoClientError::Underlying(e)
    }
}

impl From<hyper::Error> for GokoClientError {
    fn from(e: hyper::Error) -> GokoClientError {
        GokoClientError::Http(e)
    }
}

impl fmt::Display for GokoClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            GokoClientError::Underlying(ref se) => fmt::Display::fmt(se, f),
            GokoClientError::Http(ref se) => fmt::Display::fmt(se, f),
            GokoClientError::Parse(ref se) => fmt::Display::fmt(se, f),
            GokoClientError::MissingBody => f.pad("Body Missing"),
        }
    }
}

impl fmt::Debug for GokoClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            GokoClientError::Underlying(ref se) => write!(f, "Underlying({:?})", se),
            GokoClientError::Http(ref se) => write!(f, "Http({:?})", se),
            GokoClientError::Parse(ref se) => write!(f, "Underlying({:?})", se),
            GokoClientError::MissingBody => f.pad("MissingBody"),
        }
    }
}

impl Error for GokoClientError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match *self {
            GokoClientError::Underlying(ref se) => Some(se),
            GokoClientError::Http(ref se) => Some(se),
            GokoClientError::Parse(ref se) => se.source(),
            GokoClientError::MissingBody => None,
        }
    }
}
