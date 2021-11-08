use hashbrown::HashMap;
/// Smaller, serializable, version of the tracker's nodes.
use serde::{Deserialize, Serialize};

#[derive(Default, Debug, Serialize, Deserialize, Clone)]
pub struct BayesNodeSmall {
    pub kl_div: f64,
    pub mll: f64,
}

/// The stats associated to a tracker.
#[derive(Default, Debug, Serialize, Deserialize, Clone)]
pub struct BayesNodeStats {
    pub mean_kl_div: f64,
    pub mean_mll: f64,
    pub std_kl_div: f64,
    pub std_mll: f64,
    pub max_kl_div: f64,
    pub min_mll: f64,
}

/// Smaller, serializable, version of the tracker.
#[derive(Debug, Serialize, Deserialize)]
pub struct BayesCovertreeSmall {
    pub overall_tracker: BayesNodeSmall,
    pub node_trackers: HashMap<u64, BayesNodeSmall>,
    pub sequence_count: usize,
    pub window_size: usize,
}

/// Smaller, serializable, version of the tracker.
#[derive(Debug, Serialize, Deserialize)]
pub struct BayesCovertreeStats {
    pub overall_tracker: BayesNodeStats,
    pub node_trackers: HashMap<u64, BayesNodeStats>,
    pub sequence_count: usize,
    pub window_size: usize,
}
