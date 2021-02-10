//! Summaries for some label types

use hashbrown::HashMap;
use std::default::Default;
use std::iter::Iterator;

use smallvec::SmallVec;

use crate::base_traits::*;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// A summary for a small number of categories.
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct CategorySummary {
    /// Hashmap that counts how many of each instance of string there is
    pub items: SmallVec<[(i64, usize); 4]>,
}

impl Default for CategorySummary {
    fn default() -> Self {
        CategorySummary {
            items: SmallVec::new(),
        }
    }
}

impl Summary for CategorySummary {
    type Label = i64;
    fn add(&mut self, val: &i64) {
        let mut added_to_existing = false;
        for (stored_val, totals) in self.items.iter_mut() {
            if val == stored_val {
                *totals += 1;
                added_to_existing = true;
                break;
            }
        }
        if !added_to_existing {
            self.items.push((*val, 1));
        }
    }

    fn combine(&mut self, other: &CategorySummary) {
        for (val, count) in other.items.iter() {
            let mut added_to_existing = false;
            for (stored_val, totals) in self.items.iter_mut() {
                if val == stored_val {
                    *totals += count;
                    added_to_existing = true;
                    break;
                }
            }
            if !added_to_existing {
                self.items.push((*val, *count));
            }
        }
    }

    fn count(&self) -> usize {
        self.items.iter().map(|(_a, b)| b).sum()
    }
}

/// Summary of vectors
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct VecSummary {
    /// First moment, see <https://en.wikipedia.org/wiki/Moment_(mathematics)>
    pub moment1: Vec<f32>,
    /// Second moment, see <https://en.wikipedia.org/wiki/Moment_(mathematics)>
    pub moment2: Vec<f32>,
    /// The count of the number of labels included
    pub count: usize,
}

impl Summary for VecSummary {
    type Label = [f32];

    fn add(&mut self, val: &[f32]) {
        if !self.moment1.is_empty() {
            if self.moment1.len() == val.len() {
                self.moment1.iter_mut().zip(val).for_each(|(m, x)| *m += x);
                self.moment2
                    .iter_mut()
                    .zip(val)
                    .for_each(|(m, x)| *m += x * x);
                self.count += 1;
            } else {
                panic!(
                    "Combining a vec of len {:?} and of len {:?}",
                    self.moment1.len(),
                    val.len()
                );
            }
        } else {
            self.moment1.extend(val);
            self.moment2.extend(val.iter().map(|x| x * x))
        }
    }
    fn combine(&mut self, other: &VecSummary) {
        self.moment1
            .iter_mut()
            .zip(&other.moment1)
            .for_each(|(x, y)| *x += y);
        self.moment2
            .iter_mut()
            .zip(&other.moment2)
            .for_each(|(x, y)| *x += y);
        self.count += other.count;
    }

    fn count(&self) -> usize {
        self.count
    }
}

/// Summary of a bunch of underlying floats
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct FloatSummary {
    /// First moment, see <https://en.wikipedia.org/wiki/Moment_(mathematics)>
    pub moment1: f64,
    /// Second moment, see <https://en.wikipedia.org/wiki/Moment_(mathematics)>
    pub moment2: f64,
    /// The count of the number of labels included
    pub count: usize,
}

impl Summary for FloatSummary {
    type Label = f64;

    fn add(&mut self, val: &f64) {
        self.moment1 += val;
        self.moment2 += val * val;
        self.count += 1;
    }
    fn combine(&mut self, other: &FloatSummary) {
        self.moment1 += other.moment1;
        self.moment2 += other.moment2;
        self.count += other.count;
    }

    fn count(&self) -> usize {
        self.count
    }
}

/// Summary of a bunch of underlying integers, more accurate for int than the float summary
#[derive(Clone, Debug, Serialize, Deserialize, Default)]
pub struct IntSummary {
    /// First moment, see <https://en.wikipedia.org/wiki/Moment_(mathematics)>
    pub moment1: i64,
    /// Second moment, see <https://en.wikipedia.org/wiki/Moment_(mathematics)>
    pub moment2: i64,
    /// The count of the number of labels included
    pub count: usize,
}

impl Summary for IntSummary {
    type Label = i64;

    fn add(&mut self, val: &i64) {
        self.moment1 += val;
        self.moment2 += val * val;
        self.count += 1;
    }
    fn combine(&mut self, other: &IntSummary) {
        self.moment1 += other.moment1;
        self.moment2 += other.moment2;
        self.count += other.count;
    }

    fn count(&self) -> usize {
        self.count
    }
}

/// A summary for a small number of categories.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StringSummary {
    /// Hashmap that counts how many of each instance of string there is
    pub items: HashMap<String, usize>,
}

impl Default for StringSummary {
    fn default() -> Self {
        StringSummary {
            items: HashMap::new(),
        }
    }
}

impl Summary for StringSummary {
    type Label = String;
    fn add(&mut self, val: &String) {
        *self.items.entry(val.to_string()).or_insert(0) += 1;
    }

    fn combine(&mut self, other: &StringSummary) {
        for (val, count) in other.items.iter() {
            *self.items.entry(val.to_string()).or_insert(0) += count;
        }
    }

    fn count(&self) -> usize {
        self.items.values().sum()
    }
}
