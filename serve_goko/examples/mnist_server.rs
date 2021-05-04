use goko::utils::*;
use goko::CoverTreeWriter;
use pointcloud::*;
use std::path::Path;
extern crate serve_goko;
use serve_goko::parsers::MsgPackDense;
use serve_goko::http::*;
use serve_goko::core::*;
use std::sync::Arc;
use goko::plugins::discrete::prelude::GokoDirichlet;
use hyper::Server;
use log::LevelFilter;
use env_logger::Builder;

fn build_tree() -> CoverTreeWriter<DefaultLabeledCloud<L2>> {
    let file_name = "../data/mnist_complex.yml";
    let path = Path::new(file_name);
    if !path.exists() {
        panic!("{} does not exist", file_name);
    }

    cover_tree_from_labeled_yaml(&path).unwrap()
}

#[tokio::main(worker_threads = 12)]
pub async fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut builder = Builder::new();

    builder.filter_level(LevelFilter::Info).init();
    let mut ct_writer = build_tree();
    ct_writer.add_plugin::<GokoDirichlet>(GokoDirichlet {});
    ct_writer.generate_summaries();
    let goko_server = MakeGokoHttp::<_,MsgPackDense>::new(Arc::new(CoreWriter::new(ct_writer)));

    let addr = ([127, 0, 0, 1], 3031).into();

    let server = Server::bind(&addr).serve(goko_server);

    println!("Listening on http://{}", addr);

    server.await?;

    Ok(())
}
