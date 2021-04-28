use tokio::sync::oneshot;
use crate::errors::GokoClientError;
use goko::errors::GokoError;

use std::error::Error;
use std::fmt;

mod maker;
mod message;
mod service;

pub use service::GokoHttp;
pub use message::ResponseFuture;
pub use maker::MakeGokoHttp;