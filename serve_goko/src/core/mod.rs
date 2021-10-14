use goko::{CoverTreeReader, CoverTreeWriter};
use pointcloud::PointCloud;
use std::collections::HashMap;
use std::ops::Deref;
use std::sync::Arc;
use tokio::sync::RwLock;

pub(crate) mod internal_service;
use crate::api::{TrackerWorker, TrackingRequest, TrackingResponse};
use internal_service::InternalServiceOperator;

pub struct CoreWriter<D: PointCloud, T: Send + 'static> {
    pub(crate) tree: CoverTreeWriter<D>,
    pub(crate) trackers:
        Arc<RwLock<HashMap<String, InternalServiceOperator<TrackingRequest<T>, TrackingResponse>>>>,
    pub(crate) main_tracker: Arc<InternalServiceOperator<TrackingRequest<T>, TrackingResponse>>,
}

impl<D: PointCloud, T: Deref<Target = D::Point> + Send + Sync> CoreWriter<D, T> {
    pub fn new(writer: CoverTreeWriter<D>) -> Self {
        let trackers = Arc::new(RwLock::new(HashMap::new()));
        let main_tracker = Arc::new(TrackerWorker::operator(writer.reader()));
        CoreWriter {
            trackers,
            main_tracker,
            tree: writer,
        }
    }

    pub fn reader(&self) -> CoreReader<D, T> {
        let tree = self.tree.reader();
        CoreReader {
            trackers: Arc::clone(&self.trackers),
            main_tracker: Arc::clone(&self.main_tracker),
            tree,
        }
    }
}

pub struct CoreReader<D: PointCloud, T: Send + 'static> {
    pub(crate) tree: CoverTreeReader<D>,
    pub(crate) trackers:
        Arc<RwLock<HashMap<String, InternalServiceOperator<TrackingRequest<T>, TrackingResponse>>>>,
    pub(crate) main_tracker: Arc<InternalServiceOperator<TrackingRequest<T>, TrackingResponse>>,
}
