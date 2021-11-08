use super::small::*;
use super::tracker::BayesCovertree;
use core_goko::NodeAddress;
use hashbrown::HashMap;
use serde::{Deserialize, Serialize};


#[derive(Default, Debug, Serialize, Deserialize)]
struct BayesNodeMoments {
    moment1_kl_div: f64,
    moment1_mll: f64,
    moment2_kl_div: f64,
    moment2_mll: f64,
    max1_kl_div: f64,
    min1_mll: f64,
    max2_kl_div: f64,
    min2_mll: f64,
    count: usize,
}

impl BayesNodeMoments {
    fn stats(&self) -> BayesNodeStats {
        let mean_kl_div = self.moment1_kl_div / self.count as f64;
        let mean_mll = self.moment1_mll / self.count as f64;

        let var_kl_div = self.moment2_kl_div / self.count as f64 - mean_kl_div * mean_kl_div;
        let var_mll = self.moment2_mll / self.count as f64 - mean_kl_div * mean_kl_div;

        BayesNodeStats {
            mean_kl_div,
            mean_mll,
            std_kl_div: var_kl_div.max(0.0).sqrt(),
            std_mll: var_mll.max(0.0).sqrt(),
            max_kl_div: self.max1_kl_div,
            min_mll: self.min1_mll,
        }
    }

    fn stats_sans(&self, tracker: &BayesNodeSmall) -> BayesNodeStats {
        assert!(tracker.mll <= 0.0);
        let mean_kl_div = (self.moment1_kl_div - tracker.kl_div) / (self.count - 1) as f64;
        let mean_mll = (self.moment1_mll - tracker.mll) / (self.count - 1) as f64;

        let std_kl_div = ((self.moment2_kl_div - tracker.kl_div*tracker.kl_div) / (self.count - 1) as f64 - mean_kl_div * mean_kl_div).max(0.0).sqrt();
        let std_mll = ((self.moment2_mll - tracker.mll*tracker.mll)/ (self.count - 1) as f64 - mean_kl_div * mean_kl_div).max(0.0).sqrt();

        let max_kl_div = if (self.max1_kl_div - tracker.kl_div).abs() < 1e-7 {
            self.max2_kl_div
        } else {
            self.max1_kl_div
        };
        let min_mll = if (self.min1_mll - tracker.mll).abs() < 1e-7 {
            self.min2_mll
        } else {
            self.min1_mll
        };
        BayesNodeStats {
            mean_kl_div,
            mean_mll,
            std_kl_div,
            std_mll,
            max_kl_div,
            min_mll,
        }
    }

    fn update(&mut self, tracker: &BayesNodeSmall) {
        self.moment1_kl_div += tracker.kl_div;
        self.moment1_mll += tracker.mll;
        self.moment2_kl_div += tracker.kl_div * tracker.kl_div;
        self.moment2_mll += tracker.mll * tracker.mll;
        if tracker.kl_div > self.max1_kl_div {
            self.max2_kl_div = self.max1_kl_div;
            self.max1_kl_div = tracker.kl_div;
        }
        if tracker.mll < self.min1_mll {
            self.min2_mll = self.min1_mll;
            self.min1_mll = tracker.mll;
        }
        self.count += 1;
    }
}

/// Smaller, serializable, version of the tracker.
#[derive(Debug, Serialize, Deserialize)]
pub struct BayesCovertreeBaseline {
    pub overall_baseline: BayesNodeStats,
    pub node_baselines: HashMap<u64, BayesNodeStats>,
    pub window_size: usize,
}

/// Smaller, serializable, version of the tracker.
#[derive(Debug, Serialize, Deserialize)]
pub struct BayesCovertreeBaselineMoments {
    overall_baseline: BayesNodeMoments,
    node_baselines: HashMap<u64, BayesNodeMoments>,
    window_size: usize,
}

impl BayesCovertreeBaselineMoments {
    /// Creates a new baseline tracker with enough capacity that this doesn't need to reallocate
    pub fn new(window_size: usize, capacity: usize) -> BayesCovertreeBaselineMoments {
        BayesCovertreeBaselineMoments {
            overall_baseline: BayesNodeMoments::default(),
            node_baselines: HashMap::with_capacity(capacity),
            window_size,
        }
    }
    /// Update the internal baseline
    pub fn update(&mut self, tracker: &BayesCovertreeSmall) {
        tracker
            .node_trackers
            .iter()
            .for_each(|(i, v)| self.node_baselines.entry(*i).or_default().update(v));
        self.overall_baseline.update(&tracker.overall_tracker)
    }

    pub fn stats(&self) -> BayesCovertreeBaseline {
        BayesCovertreeBaseline {
            overall_baseline: self.overall_baseline.stats(),
            node_baselines: self
                .node_baselines
                .iter()
                .map(|(a, b)| (*a, b.stats()))
                .collect(),
            window_size: self.window_size,
        }
    }

    pub fn stats_sans(&self, tracker: &BayesCovertreeSmall) -> BayesCovertreeBaseline {
        BayesCovertreeBaseline {
            overall_baseline: self.overall_baseline.stats(),
            node_baselines: self
                .node_baselines
                .iter()
                .map(|(a, b)| {
                    (
                        *a,
                        b.stats_sans(
                            tracker
                                .node_trackers
                                .get(&a)
                                .unwrap_or(&BayesNodeSmall::default()),
                        ),
                    )
                })
                .collect(),
            window_size: self.window_size,
        }
    }
}

pub struct BayesCovertreeBaselineBuilder {
    tracker: BayesCovertree,
    data_points: Vec<BayesCovertreeSmall>,
    moments: BayesCovertreeBaselineMoments,
    observation_interval: usize,
    sequence_count: usize,
}

impl BayesCovertreeBaselineBuilder {
    pub fn new(
        window_size: usize,
        observation_interval: usize,
        tracker: BayesCovertree,
    ) -> Self {
        let moments = BayesCovertreeBaselineMoments::new(window_size, tracker.node_count());
        let data_points = Vec::new();

        BayesCovertreeBaselineBuilder {
            tracker,
            data_points,
            moments,
            observation_interval,
            sequence_count: 0,
        }
    }

    pub fn add_path(&mut self, trace: Vec<(NodeAddress, f32)>) {
        self.sequence_count += 1;
        self.tracker.add_path(trace);
        if self.sequence_count % self.observation_interval == 0 {
            let data_point = self.tracker.small();
            self.moments.update(&data_point);
            self.data_points.push(data_point);
        }
    }

    pub fn baseline(&self) -> BayesCovertreeBaseline {
        self.moments.stats()
    }

    pub fn loo_baselines(&self) -> Vec<BayesCovertreeBaseline> {
        self.data_points
            .iter()
            .map(|d| self.moments.stats_sans(d))
            .collect()
    }

    pub fn data_points(&self) -> &[BayesCovertreeSmall] {
        &self.data_points[..]
    }

    /// Gives you all elements in the leave one out cross validation
    pub fn loo_violators(&self) -> Vec<(u64, BayesNodeSmall, BayesNodeStats)> {
        self.data_points
            .iter()
            .map(|d| {
                d.node_trackers.iter().filter_map(|(a, v)| {
                    self.moments.node_baselines
                        .get(a)
                        .map(|m| {
                            let sans = m.stats_sans(v);
                            if (v.kl_div - sans.max_kl_div > 0.0 || v.mll - sans.min_mll < 0.0) && m.count > 1 {
                                Some((*a, v.clone(), sans))
                            } else {
                                None
                            }
                        })
                        .flatten()
                })
            })
            .flatten()
            .collect()
    }
}
