use crate::core::internal_service::*;
use goko::errors::GokoError;
use goko::plugins::discrete::tracker::BayesCovertree;
use goko::{CoverTreeReader, NodeAddress};
use pointcloud::*;
use std::ops::Deref;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::{TrackingRequest, TrackingRequestChoice, TrackingResponse};

#[derive(Deserialize, Serialize)]
pub struct TrackPointRequest<T> {
    pub point: T,
}

#[derive(Deserialize, Serialize)]
pub struct TrackPathRequest {
    pub path: Vec<(NodeAddress, f32)>,
}

#[derive(Deserialize, Serialize)]
pub struct TrackPathResponse {
    pub success: bool,
}

#[derive(Deserialize, Serialize)]
pub struct AddTrackerRequest {
    pub window_size: usize,
}
#[derive(Deserialize, Serialize)]
pub struct AddTrackerResponse {
    pub success: bool,
}

#[derive(Deserialize, Serialize)]
pub struct CurrentStatsRequest {
    pub window_size: usize,
}

#[derive(Deserialize, Serialize)]
pub struct CurrentStatsResponse {
    pub kl_div: f64,
    pub max: f64,
    pub min: f64,
    pub nz_count: u64,
    pub moment1_nz: f64,
    pub moment2_nz: f64,
    pub sequence_len: usize,
}

pub struct TrackerWorker<D: PointCloud> {
    reader: CoverTreeReader<D>,
    trackers: HashMap<usize, BayesCovertree>,
}

impl<D: PointCloud> TrackerWorker<D> {
    pub fn new(reader: CoverTreeReader<D>) -> TrackerWorker<D> {
        TrackerWorker {
            reader,
            trackers: HashMap::new(),
        }
    }

    pub(crate) fn operator<T: Deref<Target = D::Point> + Send + Sync + 'static>(
        reader: CoverTreeReader<D>,
    ) -> InternalServiceOperator<TrackingRequest<T>, TrackingResponse> {
        let worker = TrackerWorker {
            reader,
            trackers: HashMap::new(),
        };
        InternalServiceOperator::new(worker)
    }
}

impl<D: PointCloud, T: Deref<Target = D::Point> + Send + Sync>
    InternalService<TrackingRequest<T>, TrackingResponse> for TrackerWorker<D>
{
    fn process(&mut self, request: TrackingRequest<T>) -> Result<TrackingResponse, GokoError> {
        use TrackingRequestChoice::*;
        match request.request {
            TrackPoint(req) => {
                let path = self.reader.path(&req.point)?;
                for tracker in self.trackers.values_mut() {
                    tracker.add_path(path.clone());
                }

                Ok(TrackingResponse::TrackPath(TrackPathResponse {
                    success: !self.trackers.is_empty(),
                }))
            }
            TrackPath(req) => {
                for tracker in self.trackers.values_mut() {
                    tracker.add_path(req.path.clone());
                }
                Ok(TrackingResponse::TrackPath(TrackPathResponse {
                    success: true,
                }))
            }
            AddTracker(req) => {
                if self.trackers.contains_key(&req.window_size) {
                    Ok(TrackingResponse::AddTracker(AddTrackerResponse {
                        success: false,
                    }))
                } else {
                    self.trackers.insert(
                        req.window_size,
                        BayesCovertree::new(req.window_size, &self.reader),
                    );
                    Ok(TrackingResponse::AddTracker(AddTrackerResponse {
                        success: true,
                    }))
                }
            }
            CurrentStats(req) => {
                if let Some(tracker) = self.trackers.get(&req.window_size) {
                    let stats = tracker.nodes_kl_div_stats();
                    let kl_div = tracker.overall_kl_div();
                    Ok(TrackingResponse::CurrentStats(CurrentStatsResponse {
                        kl_div,
                        max: stats.max,
                        min: stats.min,
                        nz_count: stats.nz_count,
                        moment1_nz: stats.moment1_nz,
                        moment2_nz: stats.moment2_nz,
                        sequence_len: stats.sequence_len,
                    }))
                } else {
                    Ok(TrackingResponse::Unknown(
                        request.tracker_name.clone(),
                        Some(req.window_size),
                    ))
                }
            }
        }
    }
}
