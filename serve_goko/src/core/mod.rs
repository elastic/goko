use pointcloud::PointCloud;
use goko::{CoverTreeReader,CoverTreeWriter};
use std::sync::RwLock;
use crate::{GokoRequest, GokoResponse};
use std::ops::Deref;
use goko::errors::GokoError;

pub struct CoreWriter<D: PointCloud> {
    pub(crate) tree: RwLock<CoverTreeWriter<D>>,
    pub(crate) tracker: Arc<RwLock<CoverTreeWriter<D>>>,
}

impl<D: PointCloud> CoreWriter<D> {
    pub fn new(writer: CoverTreeWriter<D>) -> Self {
        
        CoreWriter {
            tree: RwLock::new(writer),
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

impl<D: PointCloud> CoreReader<D> {
    pub async fn process<P>(&mut self, request: GokoRequest<P>) -> Result<GokoResponse,GokoError>
    where P: Deref<Target = D::Point> + Send + Sync + 'static {
        match request {
            GokoRequest::Parameters(p) => p.process(self).map(|p| GokoResponse::Parameters(p)),
            GokoRequest::Knn(p) => p.process(self).map(|p| GokoResponse::Knn(p)),
            GokoRequest::RoutingKnn(p) => p.process(self).map(|p| GokoResponse::RoutingKnn(p)),
            GokoRequest::Path(p) => p.process(self).map(|p| GokoResponse::Path(p)),
            GokoRequest::Unknown(response_string, status) => {
                Ok(GokoResponse::Unknown(response_string, status))
            }
        }
    }
}