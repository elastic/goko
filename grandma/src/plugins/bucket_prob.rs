//! # Bucket probability
//!
//! A class for handling the finite probablity distribution of the children

use super::*;

use std::collections::HashMap;

/// Simple probability density function for where things go by count
///
#[derive(Debug, Clone)]
pub struct BucketProbs {
    probs: Vec<f64>,
    update: Vec<f64>,
    child_counts: HashMap<NodeAddress, usize>,
    total: usize,
}

impl BucketProbs {
    /// Creates a new empty bucket probability
    pub fn new() -> BucketProbs {
        BucketProbs {
            probs: vec![0.0],
            update: vec![0.0],
            child_counts: HashMap::new(),
            total: 0,
        }
    }

    fn index(&self, address: Option<&NodeAddress>) -> Option<usize> {
        match address {
            Some(ca) => self.child_counts.get(ca).map(|c| *c),
            None => Some(0),
        }
    }

    fn mut_index(&mut self, address: Option<NodeAddress>) -> usize {
        match address {
            None => 0,
            Some(ca) => {
                if let Some(c) = self.child_counts.get(&ca) {
                    *c
                } else {
                    let len = self.probs.len();
                    self.probs.push(0.0);
                    self.update.push(0.0);
                    self.child_counts.insert(ca, len);
                    len
                }
            }
        }
    }

    ///
    pub fn total(&self) -> usize {
        self.total
    }

    /// Updates the probs via a stochiastic gradient decent operation.
    /// The loss function is cross entropy, it does the gradient decent, then reprojects down to the affine plane.
    /// Momentum is used to make this have some form of memory
    /// If the address is not in the hashmap, the update is ignored.
    pub fn sgd_observation(
        &mut self,
        child_address: Option<&NodeAddress>,
        learning_rate: f64,
        momentum: f64,
    ) {
        if let Some(index) = self.index(child_address) {
            let mut grad: Vec<f64> = self.probs.iter().map(|p| 1.0 / (1.0 - *p)).collect();
            grad[index] = -1.0 / (self.probs[index]);
            let grad_norm = grad.iter().map(|x| x * x).fold(0.0, |x, a| x + a).sqrt();
            let grad_dot = grad.iter().map(|x| x).fold(0.0, |x, a| x + a);
            let unit_norm = (self.probs.len() as f64).sqrt();
            let proj = grad_dot / (unit_norm * grad_norm);
            self.update
                .iter_mut()
                .zip(grad)
                .for_each(|(u, g)| *u = learning_rate * (g - proj) + momentum * (*u));
            // This can result in negative values, so we check for that and step back by a bit after the update
            self.probs
                .iter_mut()
                .zip(&self.update)
                .for_each(|(p, u)| *p -= u);
            // Bring it back to a real PDF
            let mut not_updated = true;
            while not_updated {
                not_updated = false;
                let (min, bad_update) = self
                    .probs
                    .iter()
                    .zip(&self.update)
                    .min_by(|(i, _v), (j, _u)| i.partial_cmp(j).unwrap())
                    .unwrap();
                if min < &0.0 {
                    let step_back = min / (learning_rate * (bad_update - proj));
                    self.update
                        .iter_mut()
                        .for_each(|u| *u = step_back * (*u));
                    self.probs
                        .iter_mut()
                        .zip(&self.update)
                        .for_each(|(p, u)| *p -= step_back * u);
                    not_updated = false;
                }

                let (max, bad_update) = self
                    .probs
                    .iter()
                    .zip(&self.update)
                    .max_by(|(i, _v), (j, _u)| i.partial_cmp(j).unwrap())
                    .unwrap();
                if max < &0.0 {
                    let step_back = (1.0-max) / (learning_rate * (bad_update - proj));
                    self.update
                        .iter_mut()
                        .for_each(|u| *u = step_back * (*u));
                    self.probs
                        .iter_mut()
                        .zip(&self.update)
                        .for_each(|(p, u)| *p -= step_back * *u);
                    not_updated = false;
                }
            }   
            
        }
    }

    /// Adds the coverage to the key given by the child, pass none to add to the singleton pop
    pub fn add_child_pop(&mut self, child_address: Option<NodeAddress>, child_coverage: usize) {
        let index = self.mut_index(child_address);
        let target_prob = (self.probs[index] * (self.total as f64) + (child_coverage as f64))
            / ((self.total + child_coverage) as f64);
        let total_ratio = (self.total as f64) / ((self.total + child_coverage) as f64);
        self.probs.iter_mut().for_each(|p| *p *= total_ratio);
        self.probs[index] = target_prob;
        self.total += child_coverage;
    }

    /// Removes the coverage from the key given by the child, pass none to add to the singleton pop
    pub fn remove_child_pop(&mut self, child_address: Option<NodeAddress>, child_coverage: usize) {
        let index = self.mut_index(child_address);
        let cur_total = (self.probs[index] * (self.total as f64)) as usize;
        let total_ratio;
        let target_prob;
        if cur_total > child_coverage {
            total_ratio = (self.total as f64) / ((self.total - cur_total) as f64);
            target_prob = 0.0;
        } else {
            target_prob = (self.probs[index] * (self.total as f64) + (child_coverage as f64))
                / ((self.total + child_coverage) as f64);
            total_ratio = (self.total as f64) / ((self.total + child_coverage) as f64);
        }
        self.probs.iter_mut().for_each(|p| *p *= total_ratio);
        self.probs[index] = target_prob;
        self.total += cur_total;
    }

    /// Pass none if you want to test for a singleton, returns 0 if
    pub fn pdf(&self, child: Option<&NodeAddress>) -> Option<f64> {
        if self.total == 0 {
            None
        } else {
            self.index(child).map(|i| self.probs[i])
        }
    }

    /// Computes the KL divergence of two bucket probs.
    /// KL(self || other)
    /// Returns None if the support of the self is not a subset of the support of the other
    pub fn kl_divergence(&self, other: &BucketProbs) -> Option<f64> {
        if self.total == 0 || other.total == 0 {
            None
        } else {
            let mut sum: f64 = 0.0;
            let ssp = self.pdf(None).unwrap();
            let osp = other.pdf(None).unwrap();
            if ssp > 0.0 && osp > 0.0 {
                sum += ssp * (ssp.ln() - osp.ln());
            }
            if ssp > 0.0 && osp < 0.00000001 {
                return None;
            }
            for (k, v) in self.child_counts.iter() {
                if let Some(ov) = other.child_counts.get(k) {
                    let p = self.probs[*v];
                    let q = other.probs[*ov];
                    sum += p * (p.ln() - q.ln());
                } else {
                    return None;
                }
            }
            Some(sum)
        }
    }
}

impl<M: Metric> NodePlugin<M> for BucketProbs {
    fn update(&mut self, _my_node: &CoverNode<M>, _my_tree: &CoverTreeReader<M>) {}
}

/// Zero sized type that can be passed around. Equivilant to `()`
#[derive(Debug, Clone)]
pub struct BucketProbsTree {}

impl<M: Metric> TreePlugin<M> for BucketProbsTree {
    fn update(&mut self, _my_tree: &CoverTreeReader<M>) {}
}

/// Zero sized type that can be passed around. Equivilant to `()`
#[derive(Debug, Clone)]
pub struct GrandmaBucketProbs {}

/// Parent trait that make this all work. Ideally this should be included in the `TreePlugin` but rust doesn't like it.
impl<M: Metric> GrandmaPlugin<M> for GrandmaBucketProbs {
    type NodeComponent = BucketProbs;
    type TreeComponent = BucketProbsTree;
    fn node_component(
        _parameters: &Self::TreeComponent,
        my_node: &CoverNode<M>,
        my_tree: &CoverTreeReader<M>,
    ) -> Self::NodeComponent {
        let mut bucket = BucketProbs::new();

        // If we're a routing node then grab the childen's values
        if let Some((nested_scale, child_addresses)) = my_node.children() {
            my_tree.get_node_plugin_and::<Self::NodeComponent, _, _>(
                (nested_scale, *my_node.center_index()),
                |p| {
                    bucket.add_child_pop(Some((nested_scale, *my_node.center_index())), p.total());
                },
            );
            for ca in child_addresses {
                my_tree.get_node_plugin_and::<Self::NodeComponent, _, _>(*ca, |p| {
                    bucket.add_child_pop(Some(*ca), p.total());
                });
            }
            bucket.add_child_pop(None, my_node.singleton_len());
        } else {
            bucket.add_child_pop(None, my_node.singleton_len() + 1);
        }
        bucket
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    //use crate::tree::tests::build_basic_tree;

    #[test]
    fn empty_bucket_sanity_test() {
        let buckets = BucketProbs::new();
        assert_eq!(buckets.pdf(None), None);
        assert_eq!(buckets.pdf(Some(&(0, 0))), None);
        assert_eq!(buckets.kl_divergence(&buckets), None);
    }

    #[test]
    fn singleton_bucket_sanity_test() {
        let mut buckets = BucketProbs::new();
        buckets.add_child_pop(None, 5);
        assert_approx_eq!(buckets.pdf(None).unwrap(), 1.0);
        assert_approx_eq!(buckets.kl_divergence(&buckets).unwrap(), 0.0);
        assert_eq!(buckets.pdf(Some(&(0, 0))), None);
    }

    #[test]
    fn child_bucket_sanity_test() {
        let mut buckets = BucketProbs::new();
        buckets.add_child_pop(Some((0, 0)), 5);
        assert_approx_eq!(buckets.pdf(Some(&(0, 0))).unwrap(), 1.0);
        assert_approx_eq!(buckets.kl_divergence(&buckets).unwrap(), 0.0);
        assert_eq!(buckets.pdf(None).unwrap(), 0.0);
    }

    #[test]
    fn mixed_bucket_sanity_test() {
        let mut bucket1 = BucketProbs::new();
        bucket1.add_child_pop(None, 6);
        bucket1.add_child_pop(Some((0, 0)), 6);
        println!("{:?}", bucket1);

        let mut bucket2 = BucketProbs::new();
        bucket2.add_child_pop(None, 4);
        bucket2.add_child_pop(Some((0, 0)), 8);
        println!("{:?}", bucket2);

        assert_approx_eq!(bucket1.pdf(None).unwrap(), 0.5);
        assert_approx_eq!(bucket2.pdf(Some(&(0, 0))).unwrap(), 0.666666666);
        assert_approx_eq!(bucket1.kl_divergence(&bucket1).unwrap(), 0.0);

        assert_approx_eq!(bucket1.kl_divergence(&bucket2).unwrap(), 0.05889151782);
        assert_approx_eq!(bucket2.kl_divergence(&bucket1).unwrap(), 0.05663301226);
    }

    #[test]
    fn sgd_bucket_sanity_test() {
        let mut bucket1 = BucketProbs::new();
        bucket1.add_child_pop(None, 6);
        bucket1.add_child_pop(Some((0, 0)), 6);
        println!("{:?}", bucket1);

        bucket1.sgd_observation(None, 0.01, 0.0);
        println!("{:?}", bucket1);
        assert_approx_eq!(bucket1.probs[0], 0.52);
        assert_approx_eq!(bucket1.probs[1], 0.48);
    }

    #[test]
    fn sgd_bucket_mometum_test() {
        let mut bucket1 = BucketProbs::new();
        bucket1.add_child_pop(None, 6);
        bucket1.add_child_pop(Some((0, 0)), 6);
        println!("{:?}", bucket1);

        bucket1.sgd_observation(None, 0.01, 0.9);
        bucket1.sgd_observation(None, 0.01, 0.9);
        println!("{:?}", bucket1);
        assert!(bucket1.probs[0] > 0.52);
        assert!(bucket1.probs[1] < 0.48);
        assert!(false);
    }

    #[test]
    fn sgd_bucket_limit_test() {
        let mut bucket1 = BucketProbs::new();
        bucket1.add_child_pop(None, 6);
        bucket1.add_child_pop(Some((0, 0)), 6);
        println!("{:?}", bucket1);

        bucket1.sgd_observation(None, 0.5, 0.0);
        assert_approx_eq!(bucket1.probs[0], 1.0);
        assert_approx_eq!(bucket1.probs[1], 0.0);
    }
}
