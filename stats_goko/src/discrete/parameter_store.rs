use core_goko::*;
use fxhash::FxBuildHasher;
use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;
use std::iter::Iterator;

#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize, Serialize)]
pub(crate) struct DiscreteParamsIndexes {
    indexes: Vec<u64>,
    #[serde(skip_serializing)]
    hashed_indexes: Option<HashMap<u64, usize, FxBuildHasher>>,
}

impl DiscreteParamsIndexes {
    pub(crate) fn new() -> DiscreteParamsIndexes {
        DiscreteParamsIndexes {
            indexes: Vec::new(),
            hashed_indexes: None,
        }
    }

    pub(crate) fn iter(&self) -> std::slice::Iter<'_, u64> {
        self.indexes.iter()
    }

    pub(crate) fn get(&self, loc: NodeAddress) -> Option<usize> {
        if let Some(hm) = &self.hashed_indexes {
            hm.get(&loc.raw()).map(|i| *i)
        } else {
            self.indexes.binary_search(&loc.raw()).ok()
        }
    }

    fn enable_hashmap(&mut self) {
        let hashed_indexes = self
            .indexes
            .iter()
            .enumerate()
            .map(|(i, a)| (*a, i))
            .collect();
        self.hashed_indexes = Some(hashed_indexes);
        self.hashed_indexes = None;
    }

    pub(crate) fn get_or_insert(&mut self, loc: NodeAddress) -> Result<usize, usize> {
        let na: u64 = loc.raw();
        if let Some(index) = self.hashed_indexes.as_ref().map(|h| h.get(&na)).flatten() {
            Ok(*index)
        } else {
            match self.indexes.binary_search(&na) {
                Ok(index) => Ok(index),
                Err(index) => {
                    self.indexes.insert(index, na);
                    if let Some(hm) = &mut self.hashed_indexes {
                        hm.insert(na, index);
                    } else {
                        if self.indexes.len() > 100 {
                            self.enable_hashmap()
                        }
                    }
                    Err(index)
                }
            }
        }
    }

    pub(crate) fn get_from(&self, loc: NodeAddress, values: &[f64]) -> Option<f64> {
        if let Some(hm) = &self.hashed_indexes {
            hm.get(&loc.raw()).map(|i| values[*i])
        } else {
            self.indexes
                .binary_search(&loc.raw())
                .ok()
                .map(|i| values[i])
        }
    }

    pub(crate) fn replace(
        &mut self,
        loc: NodeAddress,
        count: f64,
        values: &mut Vec<f64>,
    ) -> (f64, f64) {
        let na: u64 = loc.raw();
        if let Some(index) = self.hashed_indexes.as_ref().map(|h| h.get(&na)).flatten() {
            let old_pop = values[*index];
            values[*index] = count;
            (old_pop, values[*index])
        } else {
            match self.indexes.binary_search(&na) {
                Ok(index) => {
                    let old_pop = values[index];
                    values[index] = count;
                    (old_pop, values[index])
                }
                Err(index) => {
                    self.indexes.insert(index, na);
                    if let Some(hm) = &mut self.hashed_indexes {
                        hm.insert(na, index);
                    } else {
                        if self.indexes.len() > 100 {
                            self.enable_hashmap()
                        }
                    }
                    values.insert(index, count);
                    (0.0, values[index])
                }
            }
        }
    }

    pub(crate) fn add(
        &mut self,
        loc: NodeAddress,
        count: f64,
        values: &mut Vec<f64>,
    ) -> (f64, f64) {
        let na: u64 = loc.raw();
        if let Some(index) = (self.hashed_indexes.as_ref()).map(|h| h.get(&na)).flatten() {
            let old_pop = values[*index];
            values[*index] += count;
            (old_pop, values[*index])
        } else {
            match self.indexes.binary_search(&na) {
                Ok(index) => {
                    let old_pop = values[index];
                    values[index] += count;
                    (old_pop, values[index])
                }
                Err(index) => {
                    self.indexes.insert(index, na);
                    values.insert(index, count);
                    if let Some(hm) = &mut self.hashed_indexes {
                        hm.insert(na, index);
                    } else {
                        if self.indexes.len() > 100 {
                            self.enable_hashmap()
                        }
                    }
                    (0.0, values[index])
                }
            }
        }
    }

    pub(crate) fn subtract(
        &mut self,
        loc: NodeAddress,
        count: f64,
        values: &mut Vec<f64>,
    ) -> (f64, f64) {
        let na: u64 = loc.raw();
        if let Some(hm) = &mut self.hashed_indexes {
            if let Some(index) = hm.get(&na) {
                let old_pop = values[*index];
                if old_pop > count {
                    values[*index] -= count;
                } else {
                    values[*index] = 0.0;
                }
                (old_pop, values[*index])
            } else {
                (0.0, 0.0)
            }
        } else {
            match self.indexes.binary_search(&na) {
                Ok(index) => {
                    let old_pop = values[index];
                    if old_pop > count {
                        values[index] -= count;
                    } else {
                        values[index] = 0.0;
                    }
                    (old_pop, values[index])
                }
                Err(_) => (0.0, 0.0),
            }
        }
    }
}

#[derive(Debug, Clone, Default)]
pub(crate) struct DiscreteParams {
    pub(crate) indexes: DiscreteParamsIndexes,
    pub(crate) values: Vec<f64>,
    pub(crate) total: f64,
}

impl From<&[(u64, f64)]> for DiscreteParams {
    fn from(vals: &[(u64, f64)]) -> DiscreteParams {
        let mut params = DiscreteParams::new();
        for (i, v) in vals.iter() {
            params.replace_pop(i.into(), *v);
        }
        params
    }
}

impl From<&[(NodeAddress, f64)]> for DiscreteParams {
    fn from(vals: &[(NodeAddress, f64)]) -> DiscreteParams {
        let mut params = DiscreteParams::new();
        for (i, v) in vals.iter() {
            params.replace_pop(*i, *v);
        }
        params
    }
}

pub struct DiscreteParamsIter<'a> {
    pub(crate) index_iter: std::slice::Iter<'a, u64>,
    pub(crate) value_iter: std::slice::Iter<'a, f64>,
}

impl DiscreteParams {
    /// New all 0 DiscreteParams distribution. The child counts are uninitialized
    pub(crate) fn new() -> DiscreteParams {
        DiscreteParams {
            indexes: DiscreteParamsIndexes::new(),
            values: Vec::new(),
            total: 0.0,
        }
    }

    pub(crate) fn iter(&self) -> DiscreteParamsIter {
        DiscreteParamsIter {
            index_iter: self.indexes.indexes.iter(),
            value_iter: self.values.iter(),
        }
    }

    pub(crate) fn double_iter<'b>(
        &self,
        other: &'b DiscreteParams,
    ) -> DiscreteParamsDoubleIter<'_, 'b> {
        let mut iter_a = self.iter();
        let mut iter_b = other.iter();
        let val_a = iter_a.next();
        let val_b = iter_b.next();
        DiscreteParamsDoubleIter {
            iter_a,
            iter_b,
            val_a,
            val_b,
        }
    }


    /// The total of the parameters. This is a proxy for the total count, and the "concentration" of the distribution
    pub(crate) fn total(&self) -> f64 {
        self.total
    }

    ///Gives the allocated length of the parameters
    pub(crate) fn len(&self) -> usize {
        self.values.len()
    }

    pub(crate) fn add_pop(&mut self, loc: NodeAddress, count: f64) -> (f64, f64) {
        assert!(count >= 0.0);
        self.total += count;
        self.indexes.add(loc, count, &mut self.values)
    }

    pub(crate) fn replace_pop(&mut self, loc: NodeAddress, count: f64) -> (f64, f64) {
        assert!(count >= 0.0);
        let (old, new) = self.indexes.replace(loc, count, &mut self.values);
        self.total += new - old;
        (old, new)
    }

    pub(crate) fn remove_pop(&mut self, loc: NodeAddress, count: f64) -> (f64, f64) {
        assert!(count >= 0.0);
        let (old, new) = self.indexes.subtract(loc, count, &mut self.values);
        self.total += new - old;
        (old, new)
    }

    pub(crate) fn get(&self, loc: NodeAddress) -> Option<f64> {
        self.indexes.get(loc).map(|index| self.values[index])
    }
}

impl<'a> Iterator for DiscreteParamsIter<'a> {
    type Item = (NodeAddress, f64);
    fn next(&mut self) -> Option<(NodeAddress, f64)> {
        match (self.index_iter.next(), self.value_iter.next()) {
            (Some(i), Some(v)) => Some((i.into(), *v)),
            (None, None) => None,
            _ => panic!("Should always be in lockstep"),
        }
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.index_iter.size_hint()
    }
}

pub struct DiscreteParamsDoubleIter<'a, 'b> {
    pub(crate) iter_a: DiscreteParamsIter<'a>,
    pub(crate) iter_b: DiscreteParamsIter<'b>,
    pub(crate) val_a: Option<(NodeAddress, f64)>,
    pub(crate) val_b: Option<(NodeAddress, f64)>,
}

impl<'a, 'b> Iterator for DiscreteParamsDoubleIter<'a, 'b> {
    type Item = ((NodeAddress, f64), (NodeAddress, f64));
    fn next(&mut self) -> Option<((NodeAddress, f64), (NodeAddress, f64))> {
        match (self.val_a, self.val_b) {
            (Some((a_loc, a_val)), Some((b_loc, b_val))) => match a_loc.cmp(&b_loc) {
                Ordering::Equal => {
                    self.val_a = self.iter_a.next();
                    self.val_b = self.iter_b.next();
                    Some(((a_loc, a_val), (b_loc, b_val)))
                }
                Ordering::Greater => {
                    self.val_b = self.iter_b.next();
                    Some(((b_loc, 0.0), (b_loc, b_val)))
                }
                Ordering::Less => {
                    self.val_a = self.iter_a.next();
                    Some(((a_loc, a_val), (a_loc, 0.0)))
                }
            },
            (Some((a_loc, a_val)), None) => {
                self.val_a = self.iter_a.next();
                Some(((a_loc, a_val), (a_loc, 0.0)))
            }
            (None, Some((b_loc, b_val))) => {
                self.val_b = self.iter_b.next();
                Some(((b_loc, 0.0), (b_loc, b_val)))
            }
            (None, None) => None,
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn insert_single() {
        let mut params = DiscreteParams::new();
        params.add_pop(0.into(), 5.0);
        assert_eq!(params.get(0.into()), Some(5.0));
        assert_eq!(params.get(1.into()), None);
        assert_eq!(params.total(), 5.0);
        assert_eq!(params.len(), 1);
    }

    #[test]
    fn insert_multiple() {
        let mut params = DiscreteParams::new();
        params.add_pop(0.into(), 5.0);
        params.add_pop(2.into(), 4.0);
        params.add_pop(1.into(), 3.0);
        assert_eq!(params.get(1.into()), Some(3.0));
        assert_eq!(params.get(4.into()), None);
        assert_eq!(params.total(), 12.0);
        assert_eq!(params.len(), 3);
    }

    #[test]
    fn subtract_multiple() {
        let mut params = DiscreteParams::new();
        params.add_pop(0.into(), 5.0);
        params.add_pop(2.into(), 4.0);
        params.add_pop(1.into(), 3.0);
        params.remove_pop(2.into(), 5.0);
        params.remove_pop(1.into(), 5.0);
        assert_eq!(params.get(1.into()), Some(0.0));
        assert_eq!(params.total(), 5.0);
        assert_eq!(params.len(), 3);
    }

    #[test]
    fn subtract_single() {
        let mut params = DiscreteParams::new();
        params.add_pop(0.into(), 5.0);
        params.remove_pop(0.into(), 5.0);
        assert_eq!(params.get(0.into()), Some(0.0));
        assert_eq!(params.total(), 0.0);
        assert_eq!(params.len(), 1);
    }

    #[test]
    fn total_empty() {
        let params = DiscreteParams::new();
        assert_eq!(params.total(), 0.0);
        assert_eq!(params.len(), 0);
    }

    #[test]
    fn small_iter() {
        let mut params = DiscreteParams::new();
        params.add_pop(0.into(), 5.0);
        params.add_pop(2.into(), 4.0);
        params.add_pop(1.into(), 3.0);
        let param_vec: Vec<(NodeAddress, f64)> = params.iter().collect();
        assert_eq!(
            param_vec,
            vec![
                (NodeAddress::from(0), 5.0),
                (NodeAddress::from(1), 3.0),
                (NodeAddress::from(2), 4.0)
            ]
        );
    }

    #[test]
    fn small_double_iter() {
        let mut params_1 = DiscreteParams::new();
        params_1.add_pop(0.into(), 5.0);
        params_1.add_pop(3.into(), 4.0);
        params_1.add_pop(1.into(), 3.0);
        let mut params_2 = DiscreteParams::new();
        params_2.add_pop(2.into(), 4.0);
        params_2.add_pop(1.into(), 3.0);
        let param_vec: Vec<((NodeAddress, f64), (NodeAddress, f64))> =
            params_1.double_iter(&params_2).collect();
        println!("{:?}", param_vec);
        let result_vec = vec![
            ((0.into(), 5.0), (0.into(), 0.0)),
            ((1.into(), 3.0), (1.into(), 3.0)),
            ((2.into(), 0.0), (2.into(), 4.0)),
            ((3.into(), 4.0), (3.into(), 0.0)),
        ];
        assert_eq!(param_vec.len(), result_vec.len());
        for (a, b) in param_vec.iter().zip(result_vec) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, b.1);
        }
    }
}
