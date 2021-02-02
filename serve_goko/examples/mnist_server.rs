use std::path::Path;
use goko::utils::*;
use goko::CoverTreeWriter;
use pointcloud::*;
extern crate serve_goko;
use serve_goko::*;

use hyper::service::{make_service_fn, service_fn};
use hyper::{Body, Request, Response, Server};

fn build_tree() -> CoverTreeWriter<DefaultLabeledCloud<L2>> {
    let file_name = "../data/mnist_complex.yml";
    let path = Path::new(file_name);
    if !path.exists() {
        panic!(file_name.to_owned() + &" does not exist".to_string());
    }

    cover_tree_from_labeled_yaml(&path).unwrap()
}

#[tokio::main]
pub async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    pretty_env_logger::init();
    let ct_writer = build_tree();
    
    let goko_server = GokoLayer {
        writer: ct_writer,
    };

    let service_builder = ServiceBuilder::new()
        .buffer(100)
        .concurrency_limit(10)
        .service(tree_router)
        .layer(goko_server)

    let addr = ([127, 0, 0, 1], 3030).into();

    let server = Server::bind(&addr).serve(service_builder);

    println!("Listening on http://{}", addr);

    server.await?;

    Ok(())
}