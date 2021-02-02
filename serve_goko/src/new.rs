#![deny(warnings)]

use goko::{CoverTreeReader, CoverTreeWriter};
use pointcloud::*;
use http::{Request, Response, StatusCode};
use hyper::{Body, Method};
use hyper::body::to_bytes;
use hyper::service::Service;

use std::task::Poll;
use core::task::Context;
use std::future::Future;
use std::pin::Pin;

use flate2::read::{DeflateDecoder,ZlibDecoder};
use rmp::decode;
use rmp::decode::ValueReadError;
use std::io::Read;
struct MsgPackDense;

impl MsgPackDense {
    fn decode_level(content_type: &str, mut raw: Vec<u8>) -> Vec<u8> {
        let mut new_body: Vec<u8> = Vec::new();
        match content_type {
            "zlib" => {
                let mut decoder = DeflateDecoder::new(&raw[..]);
                // read the whole file
                decoder.read_to_end(&mut new_body).unwrap();
                new_body
            },
            "gzip" => {
                let mut decoder = ZlibDecoder::new(&raw[..]);
                // read the whole file
                decoder.read_to_end(&mut new_body).unwrap();
                new_body
            },
            "base64" => {
                if raw[raw.len()-1] == b"\n"[0] {
                    raw.pop();
                }
                while raw.len() % 4 != 0 {
                    raw.push(b"="[0]);
                }
                base64::decode_config_buf(&raw[..], base64::STANDARD, &mut new_body).unwrap();
                new_body
            },
            &_ => raw,
        }
    }

    pub async fn parse(req: Request<Body>) -> Result<Vec<f32>, hyper::Error> {
        let (parts,body) = req.into_parts();
        println!("{:?}", parts);
        let mut bytes: Vec<u8> = to_bytes(body).await?.to_vec();
        
        match parts.headers.get("Content-Type") {
            Some(typestr) => {
                println!("{:?}", typestr);
                let type_str_components = typestr.to_str().unwrap().split("/");
                for encoding in type_str_components {
                    bytes = Self::decode_level(encoding,bytes);
                }
            },
            None => (),
        }

        Ok(Self::read_msgpack(&bytes[..]).unwrap())
    }

    pub fn read_msgpack(bytes: &[u8]) -> Result<Vec<f32>, ValueReadError> {
        if bytes.len() > 0 {
            let mut msgpk_slice = &bytes[..];
            let len = decode::read_array_len(&mut msgpk_slice)?;
            let mut ret: Vec<f32> = vec![];
            for _i in 0..len {
                ret.push(decode::read_f64(&mut msgpk_slice)? as f32);
            }
            Ok(ret)
        } else {
            Ok(vec![])
        }
    }
}

fn get_k_query(expr: &str) -> usize {
    lazy_static! {
        static ref RE: Regex = Regex::new(r"k=(?P<k>\d+)").unwrap();
    }
    match RE.captures(expr) {
        Some(caps) => caps["k"].parse::<usize>().unwrap(),
        None => 10,
    }
}

pub struct GokoService<D: PointCloud> {
    pub reader: CoverTreeReader<D>,
}

impl<D: PointCloud> Service<Request<Body>> for GokoService<D> {
    type Response = Response<Body>;
    type Error = hyper::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>>>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        let response = match (req.method(), req.uri().path()) {
            // Serve some instructions at /
            (&Method::GET, "/") => {
                let params = self.reader.parameters();
                Response::new(Body::from(format!(
                    "{{\"scale_base\":{},\"leaf_cutoff\":{},\"min_res_index\":{}}}",
                    params.scale_base, params.leaf_cutoff, params.min_res_index
                )))
            }
            (&Method::GET, "/knn") => {
                let k: usize = match req.uri().query() {
                    Some(k_maybe) => get_k_query(k_maybe),
                    None => 10,
                };
            }
            // The 404 Not Found route...
            _ => {
                let mut response = Response::new(Body::empty());
                *response.status_mut() = StatusCode::NOT_FOUND;
                response
            }
        };
         
        // create a response in a future.
        let fut = async {
            Ok(response)
        };

        // Return the response as an immediate future
        Box::pin(fut)
    }
}

pub struct MakeGokoService<D: PointCloud> {
    pub writer: CoverTreeWriter<D>,
}

impl<T, D: PointCloud> Service<T> for MakeGokoService<D> {
    type Response = GokoService<D>;
    type Error = hyper::Error;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _: &mut Context) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _: T) -> Self::Future {
        let reader = self.writer.reader();
        let fut = async move { Ok(GokoService { reader }) };
        Box::pin(fut)
    }
}

