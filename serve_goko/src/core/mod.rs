use pointcloud::PointCloud;
use goko::{CoverTreeReader,CoverTreeWriter};
use std::sync::{Arc,RwLock};

pub struct CoreWriter<D: PointCloud> {
    pub(crate) tree: Arc<RwLock<CoverTreeWriter<D>>>,
}

impl<D: PointCloud> CoreWriter<D> {
    pub fn new(writer: CoverTreeWriter<D>) -> Self {
        CoreWriter {
            tree: Arc::new(RwLock::new(writer)),
        }
    }

    pub fn reader(&self) -> CoreReader<D> {
        let tree = self.tree.read().unwrap().reader();
        CoreReader {
            tree,
        }
    }
}

pub struct CoreReader<D: PointCloud> {
    pub(crate) tree: CoverTreeReader<D>,
}