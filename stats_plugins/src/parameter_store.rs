use std::iter::Iterator;

#[derive(Debug, Clone, Default)]
pub(crate) struct DiscreteParams {
    params: Vec<(u64, f64)>,
    total: f64,
}

pub(crate) struct DiscreteParamsIter<'a> {
    iter: std::slice::Iter<'a, (u64, f64)>,
}

impl DiscreteParams {
    /// New all 0 DiscreteParams distribution. The child counts are uninitialized
    pub(crate) fn new() -> DiscreteParams {
        DiscreteParams {
            params: Vec::new(),
            total: 0.0,
        }
    }

    pub(crate) fn iter(&self) -> DiscreteParamsIter {
        DiscreteParamsIter {
            iter: self.params.iter(),
        }
    }

    pub(crate) fn double_iter<'b>(
        &self,
        other: &'b DiscreteParams,
    ) -> DiscreteParamsDoubleIter<'_, 'b> {
        let mut iter_a = self.params.iter();
        let mut iter_b = other.params.iter();
        let val_a = iter_a.next();
        let val_b = iter_b.next();
        DiscreteParamsDoubleIter {
            iter_a,
            iter_b,
            val_a,
            val_b,
        }
    }

    /// Produces a copy of the original parameters, but zeros them out. This is for evidence storage where you're likely to
    /// see the same distribution as before and don't want to do allocations.
    pub(crate) fn zero_copy(&self) -> DiscreteParams {
        let params = self.params.iter().map(|(a, _)| (*a, 0.0f64)).collect();
        return DiscreteParams { params, total: 0.0 };
    }

    /// Multiplies all parameters by this weight
    pub(crate) fn weight(&mut self, weight: f64) {
        self.params.iter_mut().for_each(|(_, p)| *p *= weight);
        self.total *= weight;
    }

    pub(crate) fn normalize(&mut self) {
        self.weight(1.0 / self.total);
    }

    /// The total of the parameters. This is a proxy for the total count, and the "concentration" of the distribution
    pub(crate) fn total(&self) -> f64 {
        self.total
    }

    ///Gives the allocated length of the parameters
    pub(crate) fn len(&self) -> usize {
        self.params.len()
    }

    /// True if space has been allocated for this
    pub(crate) fn is_empty(&self) -> bool {
        self.params.is_empty()
    }

    pub(crate) fn add_pop(&mut self, loc: u64, count: f64) -> f64 {
        self.total += count;
        match self.params.binary_search_by_key(&loc, |&(a, _)| a) {
            Ok(index) => {
                self.params[index].1 += count;
                self.params[index].1
            }
            Err(index) => {
                self.params.insert(index, (loc, count));
                count
            }
        }
    }

    pub(crate) fn remove_pop(&mut self, loc: u64, count: f64) -> f64 {
        if let Ok(index) = self.params.binary_search_by_key(&loc, |&(a, _)| a) {
            if self.params[index].1 < count {
                self.total -= self.params[index].1;
                self.params[index].1 = 0.0;
                0.0
            } else {
                self.total -= count;
                self.params[index].1 -= count;
                self.params[index].1
            }
        } else {
            0.0
        }
    }

    pub(crate) fn get(&self, loc: u64) -> Option<f64> {
        match self.params.binary_search_by_key(&loc, |&(a, _)| a) {
            Ok(index) => Some(self.params[index].1),
            Err(index) => None,
        }
    }
}

impl<'a> Iterator for DiscreteParamsIter<'a> {
    type Item = (u64, f64);
    fn next(&mut self) -> Option<(u64, f64)> {
        self.iter.next().map(|a| *a)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.iter.size_hint()
    }
}

pub(crate) struct DiscreteParamsDoubleIter<'a, 'b> {
    iter_a: std::slice::Iter<'a, (u64, f64)>,
    iter_b: std::slice::Iter<'b, (u64, f64)>,
    val_a: Option<&'a (u64, f64)>,
    val_b: Option<&'b (u64, f64)>,
}

impl<'a, 'b> Iterator for DiscreteParamsDoubleIter<'a, 'b> {
    type Item = ((u64, f64), (u64, f64));
    fn next(&mut self) -> Option<((u64, f64), (u64, f64))> {
        match (self.val_a, self.val_b) {
            (Some((a_loc, a_val)), Some((b_loc, b_val))) => {
                if a_loc == b_loc {
                    self.val_a = self.iter_a.next();
                    self.val_b = self.iter_b.next();
                    return Some(((*a_loc, *a_val), (*b_loc, *b_val)));
                } else if a_loc < b_loc {
                    self.val_a = self.iter_a.next();
                    return Some(((*a_loc, *a_val), (*a_loc, 0.0)));
                } else {
                    self.val_b = self.iter_b.next();
                    return Some(((*b_loc, 0.0), (*b_loc, *b_val)));
                }
            }
            (Some((a_loc, a_val)), None) => {
                self.val_a = self.iter_a.next();
                return Some(((*a_loc, *a_val), (*a_loc, 0.0)));
            }
            (None, Some((b_loc, b_val))) => {
                self.val_b = self.iter_b.next();
                return Some(((*b_loc, 0.0), (*b_loc, *b_val)));
            }
            (None, None) => {
                return None;
            }
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    #[test]
    fn insert_single() {
        let mut params = DiscreteParams::new();
        params.add_pop(0, 5.0);
        assert_eq!(params.get(0), Some(5.0));
        assert_eq!(params.get(1), None);
        assert_eq!(params.total(), 5.0);
        assert_eq!(params.len(), 1);
        assert_eq!(params.is_empty(), false);
    }

    #[test]
    fn insert_multiple() {
        let mut params = DiscreteParams::new();
        params.add_pop(0, 5.0);
        params.add_pop(2, 4.0);
        params.add_pop(1, 3.0);
        assert_eq!(params.get(1), Some(3.0));
        assert_eq!(params.get(4), None);
        assert_eq!(params.total(), 12.0);
        assert_eq!(params.len(), 3);
        assert_eq!(params.is_empty(), false);
    }

    #[test]
    fn subtract_multiple() {
        let mut params = DiscreteParams::new();
        params.add_pop(0, 5.0);
        params.add_pop(2, 4.0);
        params.add_pop(1, 3.0);
        params.remove_pop(2, 5.0);
        params.remove_pop(1, 5.0);
        assert_eq!(params.get(1), Some(0.0));
        assert_eq!(params.total(), 5.0);
        assert_eq!(params.len(), 3);
        assert_eq!(params.is_empty(), false);
    }

    #[test]
    fn subtract_single() {
        let mut params = DiscreteParams::new();
        params.add_pop(0, 5.0);
        params.remove_pop(0, 5.0);
        assert_eq!(params.get(0), Some(0.0));
        assert_eq!(params.total(), 0.0);
        assert_eq!(params.len(), 1);
        assert_eq!(params.is_empty(), false);
    }

    #[test]
    fn total_empty() {
        let params = DiscreteParams::new();
        assert_eq!(params.total(), 0.0);
        assert_eq!(params.len(), 0);
        assert_eq!(params.is_empty(), true);
    }

    #[test]
    fn zeroed_copy() {
        let mut params = DiscreteParams::new();
        params.add_pop(0, 5.0);
        params.add_pop(2, 4.0);
        params.add_pop(1, 3.0);
        let zero_params = params.zero_copy();
        assert_eq!(zero_params.get(0), Some(0.0));
        assert_eq!(zero_params.get(1), Some(0.0));
        assert_eq!(zero_params.get(2), Some(0.0));
        assert_eq!(zero_params.total(), 0.0);
        assert_eq!(zero_params.len(), 3);
        assert_eq!(zero_params.is_empty(), false);
    }

    #[test]
    fn small_iter() {
        let mut params = DiscreteParams::new();
        params.add_pop(0, 5.0);
        params.add_pop(2, 4.0);
        params.add_pop(1, 3.0);
        let param_vec: Vec<(u64, f64)> = params.iter().collect();
        assert_eq!(param_vec, vec![(0, 5.0), (1, 3.0), (2, 4.0)]);
    }

    #[test]
    fn small_double_iter() {
        let mut params_1 = DiscreteParams::new();
        params_1.add_pop(0, 5.0);
        params_1.add_pop(3, 4.0);
        params_1.add_pop(1, 3.0);
        let mut params_2 = DiscreteParams::new();
        params_2.add_pop(2, 4.0);
        params_2.add_pop(1, 3.0);
        let param_vec: Vec<((u64, f64), (u64, f64))> = params_1.double_iter(&params_2).collect();
        println!("{:?}", param_vec);
        let result_vec = vec![
            ((0, 5.0), (0, 0.0)),
            ((1, 3.0), (1, 3.0)),
            ((2, 0.0), (2, 4.0)),
            ((3, 4.0), (3, 0.0)),
        ];
        assert_eq!(param_vec.len(), result_vec.len());
        for (a, b) in param_vec.iter().zip(result_vec) {
            assert_eq!(a.0, b.0);
            assert_eq!(a.1, b.1);
        }
    }
}
