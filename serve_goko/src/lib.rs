//#![deny(warnings)]

//! # A server for Goko
//! 
//! 
//! See [`GokoRequest`] for documentation of how to query the HTTP server. 
//mod client;
//pub use client::*;
pub mod parsers;
pub mod errors;

pub mod api;
pub use api::GokoRequest;
pub use api::GokoResponse;
pub use parsers::PointParser;

pub mod http;
pub mod core;