use core_goko::*;
use std::cmp::Ordering;
use std::iter::Iterator;

#[derive(Debug, Clone, Default)]
pub(crate) struct DiscreteParamsIndexes {
    indexes: Vec<u64>,
}



impl DiscreteParamsIndexes {
    pub(crate) fn new() -> DiscreteParams {
        DiscreteParams {
            indexes: Vec::new(),
        }
    }

    pub(crate) fn get(&self, loc: NodeAddress) -> Option<usize> {
        self.params.binary_search_by_key(&loc.raw(), |&(a, _)| a)
    }

    pub(crate) fn get_or_insert(&mut self, loc: NodeAddress, count: f64) -> Result<usize, usize> {
        let na:u64 = loc.raw();
        match self.params.binary_search_by_key(&na, |&(a, _)| a) {
            Ok(index) => {
                Ok(index)
            }
            Err(index) => {
                self.params.insert(index, na);
                Err(index)
            }
        }
    }
}



