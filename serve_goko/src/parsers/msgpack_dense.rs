//! # Parser System
//! 
//! This currently isn't safe (we assume the caller is going to send requests with bodies of reasonable size), and does more allocations
//! than it strictly needs to. 

use hyper::body::to_bytes;
use hyper::{Request, header::HeaderValue, Body};

use http::header::CONTENT_TYPE;
use flate2::read::{DeflateDecoder, ZlibDecoder};
use rmp_serde;
use std::io::Read;
use crate::PointParser;

use crate::errors::*;

pub trait ParserService: Send + Sync + 'static {
    type Point;
    fn parse(&self, bytes: &[u8]) -> Result<Self::Point, GokoClientError>;
}

#[derive(Clone)]
pub struct MsgPackDense {}

pub enum Readers<R: Read> {
    Zlib(DeflateDecoder<R>),
    Gzip(ZlibDecoder<R>),
    None(R),
}

impl<R: Read> Read for Readers<R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        use Readers::*;
        match self {
            Zlib(reader) => reader.read(buf),
            Gzip(reader) => reader.read(buf),
            None(reader) => reader.read(buf),
        }
    }
}


impl PointParser for MsgPackDense {
    type Point = Vec<f32>;
    fn parse(body_buffer: &[u8], scratch_buffer: &mut Vec<u8>, request: &Request<Body>) -> Result<Self::Point, GokoClientError> {
        scratch_buffer.clear();
        let mut reader = match request.headers().get(CONTENT_TYPE) {
            Some(typestr) => {
                let token = typestr.to_str().unwrap();
                match token {
                    "zlib" => {
                        Readers::Zlib(DeflateDecoder::new(body_buffer))
                    }
                    "gzip" => {
                        Readers::Gzip(ZlibDecoder::new(body_buffer))
                    }
                    _ => {
                        return Err(GokoClientError::parse(Box::new(std::io::Error::new(std::io::ErrorKind::Other, "Unknown Content Type"))));
                    }
                }
            }
            None => Readers::None(body_buffer),
        };
        reader.read_to_end(scratch_buffer).map_err(|e| GokoClientError::parse(Box::new(e)))?;
        if scratch_buffer.len() > 0 {
            let point: Vec<f32> =
                rmp_serde::from_read_ref(scratch_buffer).map_err(|e| GokoClientError::Parse(Box::new(e)))?;
            Ok(point)
        } else {
            Err(GokoClientError::MissingBody)
        }
    }
}