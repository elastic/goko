use std::error::Error;
use std::fmt;
use goko::errors::GokoError;

//use serde::{Deserialize, Serialize};
//
pub enum GokoClientError {
    Underlying(GokoError),
    Http(hyper::Error),
    Parse(Box<dyn std::error::Error + Send>),
    MissingBody,
}

impl GokoClientError {
    pub fn parse(err: Box<dyn std::error::Error + Send>) -> Self {
        GokoClientError::Parse(err)
    }
}

impl From<GokoError> for GokoClientError {
    fn from(e: GokoError) -> GokoClientError {
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
