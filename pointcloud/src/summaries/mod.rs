use crate::errors::PointCloudResult;
use std::iter::Iterator;
use std::cmp::Eq;
use std::default::Default;

use smallvec::{SmallVec, smallvec};

use crate::base_traits::*;

#[derive(Clone,Debug)]
pub struct SmallCatSummary<T: Eq + Clone> {
    items: SmallVec<[(T,u64); 4]>,
    nones: u64,
    errors: u64,
}

impl<T: Eq + Clone> Default for SmallCatSummary<T> {
    fn default() -> Self {
        SmallCatSummary {
            items: SmallVec::new(),
            nones: 0,
            errors: 0,
        }
    }
}

impl<T: Eq + Clone> Summary for SmallCatSummary<T> {
    type Label = T;
    /// Adding a single value to the summary. When implementing please check that your value is compatible with your summary
    fn add(&mut self, v: PointCloudResult<Option<&T>>){
        if let Ok(v) = v {
            if let Some(val) = v {
                let mut added_to_existing = false;
                for (stored_val,totals) in self.items.iter_mut() {
                    if val == stored_val {
                        *totals += 1;
                        added_to_existing = true;
                        break;
                    }
                }
                if added_to_existing {
                    self.items.push((val.clone(),1));
                }
            } else {
                self.nones += 1;
            }
        } else {
            self.errors += 1;
        }
    }
    /// Merging several summaries of your data source together. This results in a summary of underlying column over
    /// the union of the indexes used to create the input summaries.
    fn combine(&mut self, other: SmallCatSummary<T>) {
        for (val,count) in other.items.iter() {
            let mut added_to_existing = false;
            for (stored_val,totals) in self.items.iter_mut() {
                if val == stored_val {
                    *totals += count;
                    added_to_existing = true;
                    break;
                }
            }
            if !added_to_existing {
                self.items.push((val.clone(),*count));
            }
        }
    }
}


#[derive(Clone,Debug,Default)]
pub struct VecSummary {
    // First moment is stored in the first half, the second in the second 
    moments: Vec<f32>,
    count: u64,
    nones: u64,
    errors: u64,
}

impl Summary for VecSummary {
    type Label = [f32];
    /// Adding a single value to the summary. When implementing please check that your value is compatible with your summary
    fn add(&mut self, v: PointCloudResult<Option<&[f32]>>){
        if let Ok(vv) = v {
            if let Some(val) = vv {
                if !self.moments.is_empty() {
                    let dim = self.moments.len();
                    if dim == val.len() * 2 {
                        self.moments.get_mut(0..dim/2).unwrap().iter_mut().zip(val).for_each(|(m,x)| *m += x);
                        self.moments.get_mut(dim/2..).unwrap().iter_mut().zip(val).for_each(|(m,x)| *m += x*x);
                        self.count += 1;
                    } else {
                        self.errors += 1;
                    }
                } else {
                    self.moments.extend(val);
                    self.moments.extend(val.iter().map(|x| x*x))
                }
            } else {
                self.nones += 1;
            }
        } else {
            self.errors += 1;
        }
    }
    /// Merging several summaries of your data source together. This results in a summary of underlying column over
    /// the union of the indexes used to create the input summaries.
    fn combine(&mut self, other: VecSummary) {
        self.moments.iter_mut().zip(other.moments).for_each(|(x,y)| *x += y);
        self.count += other.count;
        self.nones += other.nones;
        self.errors += other.errors;
    }
}
