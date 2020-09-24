//! See the paper for how this works

use crate::*;
use rayon::iter::repeatn;
use rand::thread_rng;
use crate::plugins::discrete::tracker::*;
use rand::prelude::*;

/// Trains a baseline by sampling randomly from the training set (used to create the tree)
/// This baseline is _not_ realistic.
pub struct DirichletBaseline {
    sample_rate: usize,
    sequence_len: usize,
    num_sequences: usize,
    prior_weight: f64,
    observation_weight: f64,
}

impl Default for DirichletBaseline {
    fn default() -> DirichletBaseline {
        DirichletBaseline {
            sample_rate: 100,
            sequence_len: 0,
            num_sequences: 8,
            prior_weight: 1.0,
            observation_weight: 1.0,
        }
    }
}

impl DirichletBaseline {
    /// Sets a new maxium sequence length. Set this to be the window size if you're using windows, the lenght of the test set you've got,
    /// or leave it alone as the default limit is the number of points in the training set.
    ///
    /// We sample up to this cap, linearly interpolating above this. So, the baseline produced is fairly accurate for indexes below this
    /// and unreliable above this.
    pub fn set_sequence_len(&mut self, sequence_len: usize) {
        self.sequence_len = sequence_len;
    }
    /// Sets a new count of sequences to train over, default 100. Stats for each sequence are returned.
    pub fn set_num_sequences(&mut self, num_sequences: usize) {
        self.num_sequences = num_sequences;
    }
    /// Sets a new prior weight, default 1.0. The prior is multiplied by this to increase or decrease it's importance
    pub fn set_prior_weight(&mut self, prior_weight: f64) {
        self.prior_weight = prior_weight;
    }
    /// Sets a new observation weight, default 1.0. Each discrete observation is treated as having this value.
    pub fn set_observation_weight(&mut self, observation_weight: f64) {
        self.observation_weight = observation_weight;
    }
    /// Samples at the following rate, then interpolates for sequence lengths between the following.
    pub fn set_sample_rate(&mut self, sample_rate: usize) {
        self.sample_rate = sample_rate;
    }

    /// Trains the sequences up.
    pub fn train<D: PointCloud>(
        &self,
        reader: CoverTreeReader<D>,
    ) -> GokoResult<KLDivergenceBaseline> {
        let point_indexes = reader.point_cloud().reference_indexes();
        let sequence_len = if self.sequence_len == 0 {
            point_indexes.len()
        } else {
            self.sequence_len
        };

        let results: Vec<Vec<KLDivergenceStats>> = repeatn(reader, self.num_sequences)
            .map(|reader| {
                let mut tracker = BayesCategoricalTracker::new(
                    self.prior_weight,
                    self.observation_weight,
                    0,
                    reader,
                );
                (&point_indexes[..])
                    .choose_multiple(&mut thread_rng(), sequence_len)
                    .enumerate()
                    .filter_map(|(i, pi)| {
                        tracker.add_path(tracker.reader().known_path(*pi).unwrap());
                        if i % self.sample_rate == 0 {
                            Some(tracker.kl_div_stats())
                        } else {
                            None
                        }
                    })
                    .collect()
            })
            .collect();
        let len = results[0].len();
        let mut sequence_len = Vec::with_capacity(len);
        let mut stats: Vec<KLDivergenceBaselineStats> =
            std::iter::repeat_with(KLDivergenceBaselineStats::default)
                .take(len)
                .collect();

        for i in 0..len {
            for result_vec in &results {
                stats[i].add(&result_vec[i]);
            }
            sequence_len.push(results[0][i].sequence_len);
        }
        Ok(KLDivergenceBaseline {
            num_sequences: results.len(),
            stats,
            sequence_len,
        })
    }
}

/// Tracks the non-zero (all KL divergences above 1e-10)
#[derive(Debug, Default)]
pub struct KLDivergenceBaselineStats {
    /// The maximum non-zero KL divergence
    pub max: (f64, f64),
    /// The minimum non-zero KL divergence
    pub min: (f64, f64),
    /// The number of nodes that have a non-zero divergence
    pub nz_count: (f64, f64),
    /// The first moment, use this with the `nz_count` to get the mean
    pub moment1_nz: (f64, f64),
    /// The second moment, use this with the `nz_count` and first moment to get the variance
    pub moment2_nz: (f64, f64),
}

impl KLDivergenceBaselineStats {
    pub(crate) fn add(&mut self, stats: &KLDivergenceStats) {
        self.max.0 += stats.max;
        self.max.1 += stats.max * stats.max;
        self.min.0 += stats.min;
        self.min.1 += stats.min * stats.min;
        self.nz_count.0 += stats.nz_count as f64;
        self.nz_count.1 += (stats.nz_count * stats.nz_count) as f64;
        self.moment1_nz.0 += stats.moment1_nz;
        self.moment1_nz.1 += stats.moment1_nz * stats.moment1_nz;
        self.moment2_nz.0 += stats.moment2_nz;
        self.moment2_nz.1 += stats.moment2_nz * stats.moment2_nz;
    }

    fn to_mean_var(&self, count: f64) -> KLDivergenceBaselineStats {
        let max_mean = self.max.0 / count;
        let min_mean = self.min.0 / count;
        let nz_count_mean = self.nz_count.0 / count;
        let moment1_nz_mean = self.moment1_nz.0 / count;
        let moment2_nz_mean = self.moment2_nz.0 / count;

        let max_var = self.max.1 / count - max_mean * max_mean;
        let min_var = self.min.1 / count - min_mean * min_mean;
        let nz_count_var = self.nz_count.1 / count - nz_count_mean * nz_count_mean;
        let moment1_nz_var = self.moment1_nz.1 / count - moment1_nz_mean * moment1_nz_mean;
        let moment2_nz_var = self.moment2_nz.1 / count - moment2_nz_mean * moment2_nz_mean;

        KLDivergenceBaselineStats {
            max: (max_mean, max_var),
            min: (min_mean, min_var),
            nz_count: (nz_count_mean, nz_count_var),
            moment1_nz: (moment1_nz_mean, moment1_nz_var),
            moment2_nz: (moment2_nz_mean, moment2_nz_var),
        }
    }

    fn interpolate(mut self, other: &Self, w: f64) -> Self {
        self.max.0 += w * (other.max.0 - self.max.0);
        self.max.1 += w * (other.max.1 - self.max.1);

        self.min.0 += w * (other.min.0 - self.min.0);
        self.min.1 += w * (other.min.1 - self.min.1);

        self.nz_count.0 += w * (other.nz_count.0 - self.nz_count.0);
        self.nz_count.1 += w * (other.nz_count.1 - self.nz_count.1);

        self.moment1_nz.0 += w * (other.moment1_nz.0 - self.moment1_nz.0);
        self.moment1_nz.1 += w * (other.moment1_nz.1 - self.moment1_nz.1);

        self.moment2_nz.0 += w * (other.moment2_nz.0 - self.moment2_nz.0);
        self.moment2_nz.1 += w * (other.moment2_nz.1 - self.moment2_nz.1);
        self
    }
}

/// Computing the KL div of each node's prior and posterior is expensive.
pub struct KLDivergenceBaseline {
    /// The number of sequences we're have the stats from
    pub num_sequences: usize,
    /// The lenght of the sequences we have stats for. All other values are linearly interpolated between these.
    pub sequence_len: Vec<usize>,
    /// The actual stats objects. These are stored by the moments, but are returned by (mean,var)
    pub stats: Vec<KLDivergenceBaselineStats>,
}

impl KLDivergenceBaseline {
    /// Gets the stats object that stores an approximate mean and variance of the samples.
    pub fn stats(&self, i: usize) -> KLDivergenceBaselineStats {
        match self.sequence_len.binary_search(&i) {
            Ok(index) => self.stats[index].to_mean_var(self.num_sequences as f64),
            Err(index) => {
                if index == 0 {
                    KLDivergenceBaselineStats::default()
                } else if index == self.sequence_len.len() {
                    let stats1 = self.stats[index - 2].to_mean_var(self.num_sequences as f64);
                    let stats2 = self.stats[index - 1].to_mean_var(self.num_sequences as f64);
                    let weight = ((i - self.sequence_len[index - 2]) as f64)
                        / ((self.sequence_len[index - 1] - self.sequence_len[index - 2]) as f64);
                    stats1.interpolate(&stats2, weight)
                } else {
                    let stats1 = self.stats[index - 1].to_mean_var(self.num_sequences as f64);
                    let stats2 = self.stats[index].to_mean_var(self.num_sequences as f64);
                    let weight = ((i - self.sequence_len[index - 1]) as f64)
                        / ((self.sequence_len[index] - self.sequence_len[index - 1]) as f64);
                    stats1.interpolate(&stats2, weight)
                }
            }
        }
    }
}