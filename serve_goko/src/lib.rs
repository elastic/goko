#![feature(generic_associated_types)]

use goko::{CoverTreeReader, CoverTreeWriter};
use http::{Request, Response, StatusCode};
use hyper::body::to_bytes;
use hyper::{header::HeaderValue, Body, Method};
use pointcloud::*;
use regex::Regex;

use tower::Service;
use tower::load::Load;

use flate2::read::{DeflateDecoder, ZlibDecoder};
use rmp_serde;
use std::io::Read;
use lazy_static::lazy_static;

use tokio::sync::{mpsc,oneshot};
use std::fmt;
use std::error::Error;
use goko::errors::GokoError;
use std::convert::Infallible;

use std::task::Poll;
use core::task::Context;
use std::future::Future;
use std::pin::Pin;

use std::sync::{Arc, atomic};

use serde::{Deserialize, Serialize};
use std::marker::PhantomData;
use std::ops::Deref;

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



fn decode_level(content_type: &str, mut raw: Vec<u8>) -> Vec<u8> {
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

pub async fn parse_body<P: ParserService>(
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


pub async fn parse_request<P: ParserService>(
    parser: &P,
    request: Request<Body>,
) -> Result<GokoRequest<P::Point>, hyper::Error> {
    let (parts, body) = request.into_parts();
    match (parts.method, parts.uri.path()) {
        // Serve some instructions at /
        (Method::GET, "/") => Ok(GokoRequest::Parameters),
        (Method::GET, "/knn") => {
            lazy_static! {
                static ref RE: Regex = Regex::new(r"k=(?P<k>\d+)").unwrap();
            }

            let k: usize = match parts.uri.query().map(|s| RE.captures(s)).flatten() {
                Some(caps) => caps["k"].parse::<usize>().unwrap(),
                None => 10,
            };
            let vec = parse_body(parser, parts.headers.get("Content-Type"), body).await?;
            Ok(GokoRequest::Knn {
                vec,
                k,
            })
        }
        // The 404 Not Found route...
        _ => Ok(GokoRequest::Unknown),
    }
}

pub fn process<'a, D: PointCloud, T: Deref<Target = D::Point> + Send + Sync>(request: GokoRequest<T>, reader: &CoverTreeReader<D>) -> Response<Body> {
    use GokoRequest::*;
    match request {
        Parameters => {
            let params = reader.parameters();
            Response::new(Body::from(format!(
                "{{\"scale_base\":{},\"leaf_cutoff\":{},\"min_res_index\":{}}}",
                params.scale_base, params.leaf_cutoff, params.min_res_index
            )))
        }
        Knn {
            vec,
            k,
        } => {
            match reader.knn(&vec,k) {
                Ok(knn) => {
                    Response::new(Body::from(serde_json::to_string(&knn).unwrap()))
                }
                Err(e) => {
                    Response::new(Body::from(format!("{:?}",e)))
                }
            }
            
        }
        Unknown => {
            let mut response = Response::new(Body::empty());
            *response.status_mut() = StatusCode::NOT_FOUND;
            response
        }
    }
}

/// A summary for a small number of categories.
#[derive(Deserialize, Serialize)]
pub enum GokoRequest<T> {
    Parameters,
    Knn {
        k: usize,
        vec: T,
    },
    Unknown,
}

type ResponseSender = oneshot::Sender<Result<Response<Body>,hyper::Error>>;
type ResponseReciever = oneshot::Receiver<Result<Response<Body>,hyper::Error>>;

type RequestSender = mpsc::UnboundedSender<(Request<Body>,ResponseSender)>;
type RequestReciever = mpsc::UnboundedReceiver<(Request<Body>,ResponseSender)>;

pub struct GokoClient<D> {
    in_flight: Arc<atomic::AtomicU32>,
    request_snd: RequestSender,
    pointcloud: PhantomData<D>,
}

impl<D: PointCloud> GokoClient<D>
    where 
    D: PointCloud,
     {
    pub fn new<'a,P>(parser: P, reader: CoverTreeReader<D>) -> GokoClient<D>
    where
    P: ParserService,
    P::Point: Deref<Target = D::Point> + Send + Sync,
     {
        let (request_snd, mut request_rcv): (RequestSender, RequestReciever) =  mpsc::unbounded_channel();
        tokio::spawn(async move {
            while let Some((request,response_tx)) = request_rcv.recv().await {
                match parse_request(&parser,request).await {
                    Ok(goko_request) => response_tx.send(Ok(process(goko_request, &reader))),
                    Err(e) => response_tx.send(Err(e)),
                }.unwrap();
            };
        });
        let in_flight = Arc::new(atomic::AtomicU32::new(0));
        GokoClient {
            in_flight,
            request_snd,
            pointcloud: PhantomData,
        }
    }
}

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

impl<T> From<mpsc::error::SendError<(Request<Body>,T)>> for GokoClientError {
    fn from(e: mpsc::error::SendError<(Request<Body>,T)>) -> GokoClientError {
        GokoClientError::SendError(e.0.0)
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

impl<D> Service<Request<Body>> for GokoClient<D> {
    type Response = Response<Body>;
    type Error = GokoClientError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _: &mut Context) -> Poll<Result<(), Self::Error>> {
        if self.request_snd.is_closed() {
            Poll::Ready(Err(GokoClientError::ClientDropped))
        } else {
            Poll::Ready(Ok(()))
        }
    }

    fn call(&mut self, req: Request<Body>) -> Self::Future {
        let in_flight_req = Arc::clone(&self.in_flight);
        self.in_flight.fetch_add(1, atomic::Ordering::SeqCst);
        let (tx,rx): (ResponseSender,ResponseReciever) = oneshot::channel();
        match self.request_snd.send((req,tx)) {
            Ok(_) => {
                Box::pin(async move { 
                    let res = rx.await.unwrap().map_err(|e| e.into());
                    in_flight_req.fetch_sub(1, atomic::Ordering::SeqCst);
                    res
                 })
            },
            Err(e) => {
                Box::pin(async {Err(e.into())})
            }
        }
        
    }
}

impl<D> Load for GokoClient<D> {
    type Metric = u32;
    fn load(&self) -> Self::Metric {
        self.in_flight.load(atomic::Ordering::SeqCst)
    }
}

pub struct MakeGokoService<D: PointCloud, P> {
    writer: CoverTreeWriter<D>,
    parser: P,
}

impl<'a, D,P> MakeGokoService<D, P> 
    where 
    D: PointCloud,
    P: ParserService,
    P::Point: Deref<Target = D::Point> + Send + Sync,
{
    pub fn new(parser: P, writer: CoverTreeWriter<D>) -> MakeGokoService<D, P>
    {
        MakeGokoService {
            parser,
            writer,
        }
    }
}

impl<'a, P, D, T> Service<T> for MakeGokoService<D, P>
    where 
        D: PointCloud,
        P: ParserService + Clone,
        P::Point: Deref<Target = D::Point> + Send + Sync, 
    {
    type Response = GokoClient<D>;
    type Error = Infallible;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _: &mut Context) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, _: T) -> Self::Future {
        let reader = self.writer.reader();
        let parser = self.parser.clone();
        Box::pin(async {Ok(GokoClient::new(parser, reader))})
    }
}

