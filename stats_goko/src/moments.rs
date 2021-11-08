use serde::{Deserialize, Serialize};
/// Tracks the non-zero KL div (all KL divergences above 1e-10)
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct Moments {
    /// The number of nodes that have a non-zero divergence
    pub count: u64,
    /// The first moment, use this with the `count` to get the mean
    pub moment1: f64,
    /// The second moment, use this with the `count` and first moment to get the variance
    pub moment2: f64,
}

impl Moments {
    pub fn mean(&self) -> f64 {
        if self.count > 0 {
            self.moment1/self.count as f64
        } else {
            0.0
        }
    }

    pub fn var(&self) -> f64 {
        if self.count > 0 {
            let m = self.moment1/self.count as f64;
            self.moment2/self.count as f64 - m*m
        } else {
            0.0
        }
    }

    pub fn add(&mut self, new_stat: f64) {
        if new_stat > 1e-7 {
            self.count += 1;
            self.moment1 += new_stat;
            self.moment2 += new_stat*new_stat;
        }
    }
    pub fn remove(&mut self, old_stat: f64) {
        if old_stat > 1e-7 {
            self.count -= 1;
            self.moment1 -= old_stat;
            self.moment2 -= old_stat*old_stat;
            if self.moment1 < 0.0 {
                self.moment1 = 0.0;
            }
            if self.moment2 < 0.0 {
                self.moment2 = 0.0;
            }
        }
    }
}