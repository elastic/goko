//! # Bucket probability
//!
//! A class for handling the finite probablity distribution of the children

use super::*;
use std::fmt;

use std::collections::HashMap;

/// Simple probability density function for where things go by count
///
#[derive(Clone)]
pub struct BucketProbs {
    probs: Vec<f64>,
    update: Vec<f64>,
    child_counts: HashMap<NodeAddress, usize>,
    total: usize,
}

impl fmt::Debug for BucketProbs {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "BucketProbs {{ probs: {:?}, update: {:?}}}",
            self.probs, self.update,
        )
    }
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

    fn valid(&self) -> bool {
        let mut total = 0.0;
        for p in &self.probs {
            if !(0.0 <= *p && *p <= 1.0) {
                return false;
            }
            total += p
        }
        0 < self.probs.len() && 0.999 < total && total < 1.001
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

    fn prob_to_reals(prob: f64) -> f64 {
        0.5 * ((2.0 * prob - 1.0).atanh())
    }

    fn reals_to_prob(real: f64) -> f64 {
        (0.5 * ((2.0 * real).tanh() + 1.0))
    }

    /// Computes the gradient of a vector in the tangent space of the probability space.
    fn decend_update_vec(&mut self, index: usize, learning_rate: f64, momentum: f64) {
        let mut grad: Vec<f64> = self
            .probs
            .iter()
            .map(|p| {
                if p != &0.0 && p != &1.0 {
                    1.0 / (1.0 - *p)
                } else {
                    0.0
                }
            })
            .collect();
        if grad[index] != 0.0 && grad[index] != 1.0 {
            grad[index] = -1.0 / grad[index];
        } else {
            grad[index] = -10.0;
        }
        self.update.iter_mut().zip(&grad).for_each(|(u, g)| {
            if *g != 0.0 {
                let pu = learning_rate * g + momentum * (*u);
                if pu > 5.0 {
                    *u = 5.0;
                } else {
                    if pu < -5.0 {
                        *u = -5.0;
                    } else {
                        *u = pu;
                    }
                }
            }
        });
        // Make the update vec sum to zero, so when we add it to our PDF it sums to one
        let update_dot = self.update.iter().map(|x| x).fold(0.0, |x, a| x + a);
        let unit_norm = (self.probs.len() as f64);
        let proj = update_dot / unit_norm;
        self.update.iter_mut().for_each(|u| {
            *u = *u - proj;
        });
    }

    fn calc_scale(p:f64,u:f64) -> f64 {
        if 0.0 != u {
            let r = Self::prob_to_reals(p);
            let pr = Self::reals_to_prob(r - u);
            let correction = (pr - p).abs();
            //if p.is_nan() || r.is_nan() || pr.is_nan() || u.is_nan() || correction.is_nan() {
            //    panic!("p:{:?},r:{},pr:{},u:{},correction:{}",p,r,pr,u,correction);
            //}
            (correction / u).abs()
        } else {
            1.0
        }
    }

    /// Updates the probs via a stochiastic gradient decent operation.
    /// The loss function is cross entropy, it does the gradient decent, then reprojects down to the affine plane.
    /// The gradient is calculated wrt a metric inspired by a poicare disk. Sorta.
    /// If the address is not in the hashmap, the update is ignored.
    ///
    pub fn sgd_observation(
        &mut self,
        child_address: Option<&NodeAddress>,
        learning_rate: f64,
        momentum: f64,
    ) {
        let good_probs = self.probs.clone();
        let good_update = self.update.clone();
        if let Some(index) = self.index(child_address) {
            self.decend_update_vec(index, learning_rate, momentum);
            // We have the component of the tangent space we want to apply.
            // We could solve for the geodesic or work out a safe distance for our update vector.
            // This calculates an approximation of the length of the geodesic from the current prob vector 
            // in the direction and magnitude of the update vector.
            // The update vector's components sum to zero, we just need to make sure no values exceed 1 or 0.
            let min_metric_scale = self
                .probs
                .iter()
                .zip(&self.update)
                .map(|(p,u)|Self::calc_scale(*p,*u))
                .min_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap();
            // This can result in negative values, so we check for that and step back by a bit after the update
            self.probs
                .iter_mut()
                .zip(&self.update)
                .for_each(|(p, u)| *p -= u * min_metric_scale);
            if !self.valid() {
                let scaling: Vec<f64> = good_probs
                .iter()
                .zip(&self.update)
                .map(|(p, u)| {
                    if 0.0 != *u {
                        let r = Self::prob_to_reals(*p);
                        let pr = Self::reals_to_prob(r - u);
                        let correction = (pr - *p).abs();
                        println!("p:{:?},r:{},pr:{},u:{},correction:{}",p,r,pr,u,correction);
                        (correction / u).abs()
                    } else {
                        1.0
                    }
                }).collect();
                println!("Good probs {:?}", good_probs);
                println!("Good update {:?}", good_update);
                println!("Observation: {:?}, scale:{}", index,min_metric_scale);
                println!("Bad probs {:?}", self.probs);
                println!("Bad update {:?}", self.update);
                println!("Bad scales {:?}", scaling);
                panic!("INVALID PROB");
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
    fn space_conversion() {
        let prob = 0.5;
        let real = 0.0;
        assert_approx_eq!(prob, BucketProbs::reals_to_prob(real));
        assert_approx_eq!(real, BucketProbs::prob_to_reals(prob));
        let test_arr = [0.2, 0.542, 0.734];
        for t in test_arr.iter() {
            assert_approx_eq!(
                t,
                BucketProbs::reals_to_prob(BucketProbs::prob_to_reals(*t))
            );
        }
    }

    #[test]
    fn scale_conversion() {
        for p in 0..1000 {
            for u in 0..1000 {
                let pf = ((p+1) as f64)/1000.0;
                let uf = ((u+1) as f64)/1000.0;
                let scale = BucketProbs::calc_scale(pf,uf);
                let final_prob = pf - uf * scale;
                if !(0.0 <= final_prob && final_prob <= 1.0 ){
                    println!("p:{},u{},s:{} final_prob:{}",pf,uf,scale,final_prob);
                    assert!(false,"look!");
                }
                
            }
        }
    }

    #[test]
    fn empty_bucket_sanity_test() {
        let buckets = BucketProbs::new();
        assert_eq!(buckets.pdf(None), None);
        assert_eq!(buckets.pdf(Some(&(0, 0))), None);
        assert_eq!(buckets.kl_divergence(&buckets), None);
        assert!(!buckets.valid());
    }

    #[test]
    fn singleton_bucket_sanity_test() {
        let mut buckets = BucketProbs::new();
        buckets.add_child_pop(None, 5);
        assert_approx_eq!(buckets.pdf(None).unwrap(), 1.0);
        assert_approx_eq!(buckets.kl_divergence(&buckets).unwrap(), 0.0);
        assert_eq!(buckets.pdf(Some(&(0, 0))), None);
        assert!(buckets.valid());
    }

    #[test]
    fn child_bucket_sanity_test() {
        let mut buckets = BucketProbs::new();
        buckets.add_child_pop(Some((0, 0)), 5);
        assert_approx_eq!(buckets.pdf(Some(&(0, 0))).unwrap(), 1.0);
        assert_approx_eq!(buckets.kl_divergence(&buckets).unwrap(), 0.0);
        assert_eq!(buckets.pdf(None).unwrap(), 0.0);
        assert!(buckets.valid());
    }

    #[test]
    fn mixed_bucket_sanity_test() {
        let mut bucket1 = BucketProbs::new();
        bucket1.add_child_pop(None, 6);
        bucket1.add_child_pop(Some((0, 0)), 6);
        println!("{:?}", bucket1);
        assert!(bucket1.valid());

        let mut bucket2 = BucketProbs::new();
        bucket2.add_child_pop(None, 4);
        bucket2.add_child_pop(Some((0, 0)), 8);
        println!("{:?}", bucket2);
        assert!(bucket2.valid());

        assert_approx_eq!(bucket1.pdf(None).unwrap(), 0.5);
        assert_approx_eq!(bucket2.pdf(Some(&(0, 0))).unwrap(), 0.666666666);
        assert_approx_eq!(bucket1.kl_divergence(&bucket1).unwrap(), 0.0);
        assert!(bucket2.valid());

        assert_approx_eq!(bucket1.kl_divergence(&bucket2).unwrap(), 0.05889151782);
        assert_approx_eq!(bucket2.kl_divergence(&bucket1).unwrap(), 0.05663301226);
        assert!(bucket1.valid());
    }

    #[test]
    fn sgd_bucket_sanity_test() {
        let mut bucket1 = BucketProbs::new();
        bucket1.add_child_pop(None, 6);
        bucket1.add_child_pop(Some((0, 0)), 6);
        println!("{:?}", bucket1);

        bucket1.sgd_observation(None, 0.01, 0.0);
        println!("{:?}", bucket1);
        assert!(bucket1.probs[0] > 0.51);
        assert!(bucket1.probs[1] < 0.49);
        assert!(bucket1.valid());
    }

    #[test]
    fn sgd_bucket_mometum_test() {
        let mut bucket1 = BucketProbs::new();
        bucket1.add_child_pop(None, 11);
        bucket1.add_child_pop(Some((0, 0)), 1);
        println!("{:?}", bucket1);

        let init_probs = bucket1.probs.clone();

        bucket1.sgd_observation(None, 0.01, 0.9);
        println!("{:?}", bucket1);
        let init_diff: Vec<f64> = init_probs
            .iter()
            .zip(&bucket1.probs)
            .map(|(a, b)| (a - b).abs())
            .collect();
        let second_probs = bucket1.probs.clone();
        bucket1.sgd_observation(None, 0.01, 0.9);
        let second_diff: Vec<f64> = second_probs
            .iter()
            .zip(&bucket1.probs)
            .map(|(a, b)| (a - b).abs())
            .collect();

        println!("{:?}", bucket1);
        assert!(
            BucketProbs::prob_to_reals(second_diff[0]) > BucketProbs::prob_to_reals(init_diff[0])
        );
        assert!(
            BucketProbs::prob_to_reals(second_diff[1]) > BucketProbs::prob_to_reals(init_diff[1])
        );
        assert!(bucket1.valid());
    }

    #[test]
    fn sgd_bucket_limit_test() {
        let mut bucket1 = BucketProbs::new();
        bucket1.add_child_pop(None, 6);
        bucket1.add_child_pop(Some((0, 0)), 6);
        println!("{:?}", bucket1);

        bucket1.sgd_observation(None, 0.5, 0.0);
        assert!(bucket1.valid());
    }
}
