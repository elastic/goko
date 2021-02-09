use hyper::body::to_bytes;
use hyper::{header::HeaderValue, Body};

use flate2::read::{DeflateDecoder, ZlibDecoder};
use rmp_serde;
use std::io::Read;

use crate::errors::*;

pub trait ParserService: Send + Sync + 'static {
    type Point;
    fn parse(&self, bytes: &[u8]) -> Result<Self::Point, GokoClientError>;
}

#[derive(Clone)]
pub struct MsgPackDense {}

impl MsgPackDense {
    pub fn new() -> Self {
        MsgPackDense {}
    }
}

impl ParserService for MsgPackDense {
    type Point = Vec<f32>;
    fn parse(&self, bytes: &[u8]) -> Result<Vec<f32>, GokoClientError> {
        if bytes.len() > 0 {
            let point: Vec<f32> = rmp_serde::from_read_ref(bytes).map_err(|e| GokoClientError::Parse(Box::new(e)))?;
            Ok(point)
        } else {
            Err(GokoClientError::MissingBody)
        }
    }
}

pub(crate) fn decode_level(content_type: &str, mut raw: Vec<u8>) -> Vec<u8> {
    let mut new_body: Vec<u8> = Vec::new();
    match content_type {
        "zlib" => {
            let mut decoder = DeflateDecoder::new(&raw[..]);
            // read the whole file
            decoder.read_to_end(&mut new_body).unwrap();
            new_body
        }
        "gzip" => {
            let mut decoder = ZlibDecoder::new(&raw[..]);
            // read the whole file
            decoder.read_to_end(&mut new_body).unwrap();
            new_body
        }
        "base64" => {
            if raw[raw.len() - 1] == b"\n"[0] {
                raw.pop();
            }
            while raw.len() % 4 != 0 {
                raw.push(b"="[0]);
            }
            base64::decode_config_buf(&raw[..], base64::STANDARD, &mut new_body).unwrap();
            new_body
        }
        &_ => raw,
    }
}

pub(crate) async fn parse_body<P: ParserService>(
    parser: &P,
    content_type: Option<&HeaderValue>,
    body: Body,
) -> Result<P::Point, hyper::Error> {
    let mut bytes: Vec<u8> = to_bytes(body).await?.to_vec();

    match content_type {
        Some(typestr) => {
            let type_str_components = typestr.to_str().unwrap().split("/");
            for encoding in type_str_components {
                bytes = decode_level(encoding, bytes);
            }
        }
        None => (),
    }

    Ok(parser.parse(&bytes[..]).unwrap())
}
