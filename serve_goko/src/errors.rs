use http::Request;
use hyper::Body;

use std::error::Error;
use std::fmt;
use tokio::sync::mpsc;

//use serde::{Deserialize, Serialize};
//
//#[derive(Deserialize, Serialize)]
pub enum GokoClientError {
    SendError(Request<Body>),
    ClientDropped,
    Underlying(hyper::Error),
    Parse(Box<dyn std::error::Error + Send + Sync>),
    MissingBody,
}

impl From<hyper::Error> for GokoClientError {
    fn from(e: hyper::Error) -> GokoClientError {
        GokoClientError::Underlying(e)
    }
}

impl<T> From<mpsc::error::SendError<(Request<Body>, T)>> for GokoClientError {
    fn from(e: mpsc::error::SendError<(Request<Body>, T)>) -> GokoClientError {
        GokoClientError::SendError(e.0 .0)
    }
}

impl fmt::Display for GokoClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            GokoClientError::Underlying(ref se) => fmt::Display::fmt(se, f),
            GokoClientError::Parse(ref se) => fmt::Display::fmt(se, f),
            GokoClientError::ClientDropped => f.pad("Client was dropped"),
            GokoClientError::MissingBody => f.pad("Body Missing"),
            GokoClientError::SendError(_) => f.pad("Send Failed"),
        }
    }
}

impl fmt::Debug for GokoClientError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match *self {
            GokoClientError::Underlying(ref se) => write!(f, "Underlying({:?})", se),
            GokoClientError::Parse(ref se) => write!(f, "Underlying({:?})", se),
            GokoClientError::ClientDropped => f.pad("ClientDropped"),
            GokoClientError::MissingBody => f.pad("MissingBody"),
            GokoClientError::SendError(_) => f.pad("SendError"),
        }
    }
}

impl Error for GokoClientError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        match *self {
            GokoClientError::Underlying(ref se) => Some(se),
            GokoClientError::Parse(ref se) => se.source(),
            GokoClientError::ClientDropped => None,
            GokoClientError::MissingBody => None,
            GokoClientError::SendError(_) => None,
        }
    }
}
