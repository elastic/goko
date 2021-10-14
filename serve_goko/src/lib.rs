//#![deny(warnings)]

//! # A server for Goko
//!
//!
//! See [`GokoRequest`] for documentation of how to query the HTTP server.
//mod client;
//pub use client::*;
pub mod errors;
pub mod parsers;

pub mod api;
pub use api::GokoRequest;
pub use api::GokoResponse;
pub use parsers::PointParser;

pub mod core;
pub mod http;
