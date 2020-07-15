//! # Categorical Distribution
//!
//! Simple probability distribution that enables you to simulated the rough
//! distribution of data in the tree.

use crate::node::CoverNode;
use crate::plugins::distributions::DiscreteDistribution;
use crate::plugins::*;
use crate::tree::CoverTreeReader;

/// Simple probability density function for where things go by count
/// Stored as a flat vector in the order of the node addresses.
#[derive(Debug, Clone, Default)]
pub struct Categorical {
    child_counts: Vec<(NodeAddress, f64)>,
    singleton_count: f64,
}

impl DiscreteDistribution for Categorical {
    /// Pass none if you want to test for a singleton, returns 0 if
    fn ln_prob(&self, loc: Option<&NodeAddress>) -> Option<f64> {
        if self.total() > 0.0 {
            let ax = match loc {
                Some(ca) => self
                    .child_counts
                    .binary_search_by_key(&ca, |(a, _)| a)
                    .map(|i| self.child_counts[i].1)
                    .unwrap_or(0.0),
                None => self.singleton_count,
            };
            Some(ax.ln() - self.total().ln())
        } else {
            None
        }
    }

    /// Computes the KL divergence of two bucket probs.
    /// KL(self || other)
    /// Returns None if the support of the self is not a subset of the support of the other
    fn kl_divergence(&self, other: &Categorical) -> Option<f64> {
        let my_total = self.total();
        let other_total = other.total();
        if my_total == 0.0 || other_total == 0.0 {
            None
        } else {
            let ln_total = my_total.ln() - other_total.ln();
            let mut sum: f64 = 0.0;
            if self.singleton_count > 0.0 && other.singleton_count > 0.0 {
                sum += (self.singleton_count / my_total)
                    * (self.singleton_count.ln() - other.singleton_count.ln() - ln_total);
            }
            for ((ca, ca_count), (other_ca, other_ca_count)) in
                self.child_counts.iter().zip(other.child_counts.iter())
            {
                assert_eq!(ca, other_ca);
                sum += (ca_count / my_total) * (ca_count.ln() - other_ca_count.ln() - ln_total);
            }
            Some(sum)
        }
    }
}

impl Categorical {
    /// Creates a new empty bucket probability
    pub fn new() -> Categorical {
        Categorical {
            child_counts: Vec::new(),
            singleton_count: 0.0,
        }
    }

    /// Total input to this categorical distribution.
    pub fn total(&self) -> f64 {
        self.singleton_count
            + self
                .child_counts
                .iter()
                .map(|(_, c)| c)
                .fold(0.0, |x, a| x + a)
    }

    fn add_child_pop(&mut self, loc: Option<NodeAddress>, count: f64) {
        match loc {
            Some(ca) => match self.child_counts.binary_search_by_key(&ca, |&(a, _)| a) {
                Ok(index) => self.child_counts[index].1 += count,
                Err(index) => self.child_counts.insert(index, (ca, count)),
            },
            None => self.singleton_count += count,
        }
    }

    fn remove_child_pop(&mut self, loc: Option<NodeAddress>, count: f64) {
        match loc {
            Some(ca) => {
                if let Ok(index) = self.child_counts.binary_search_by_key(&ca, |&(a, _)| a) {
                    if self.child_counts[index].1 < count {
                        self.child_counts[index].1 = 0.0;
                    } else {
                        self.child_counts[index].1 -= count;
                    }
                }
            }
            None => {
                if self.singleton_count < count as f64 {
                    self.singleton_count = 0.0;
                } else {
                    self.singleton_count -= count as f64;
                }
            }
        }
    }
}

impl<D: PointCloud> NodePlugin<D> for Categorical {
    fn update(&mut self, _my_node: &CoverNode<D>, _my_tree: &CoverTreeReader<D>) {}
}

/// Zero sized type that can be passed around. Equivilant to `()`
#[derive(Debug, Clone)]
pub struct CategoricalTree {}

impl<D: PointCloud> TreePlugin<D> for CategoricalTree {
    fn update(&mut self, _my_tree: &CoverTreeReader<D>) {}
}

/// Zero sized type that can be passed around. Equivilant to `()`
#[derive(Debug, Clone)]
pub struct GokoCategorical {}

/// Parent trait that make this all work. Ideally this should be included in the `TreePlugin` but rust doesn't like it.
impl<D: PointCloud> GokoPlugin<D> for GokoCategorical {
    type NodeComponent = Categorical;
    type TreeComponent = CategoricalTree;
    fn node_component(
        _parameters: &Self::TreeComponent,
        my_node: &CoverNode<D>,
        my_tree: &CoverTreeReader<D>,
    ) -> Self::NodeComponent {
        let mut bucket = Categorical::new();

        // If we're a routing node then grab the childen's values
        if let Some((nested_scale, child_addresses)) = my_node.children() {
            my_tree.get_node_plugin_and::<Self::NodeComponent, _, _>(
                (nested_scale, *my_node.center_index()),
                |p| {
                    bucket.add_child_pop(
                        Some((nested_scale, *my_node.center_index())),
                        p.total() as f64,
                    );
                },
            );
            for ca in child_addresses {
                my_tree.get_node_plugin_and::<Self::NodeComponent, _, _>(*ca, |p| {
                    bucket.add_child_pop(Some(*ca), p.total() as f64);
                });
            }
            bucket.add_child_pop(None, my_node.singletons_len() as f64);
        } else {
            bucket.add_child_pop(None, my_node.singletons_len() as f64 + 1.0);
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
        let buckets = Categorical::new();
        assert_eq!(buckets.ln_prob(None), None);
        assert_eq!(buckets.ln_prob(Some(&(0, 0))), None);
        assert_eq!(buckets.kl_divergence(&buckets), None)
    }

    #[test]
    fn singleton_bucket_sanity_test() {
        let mut buckets = Categorical::new();
        buckets.add_child_pop(None, 5.0);
        assert_approx_eq!(buckets.ln_prob(None).unwrap(), 0.0);
        assert_approx_eq!(buckets.kl_divergence(&buckets).unwrap(), 0.0);
        assert_eq!(buckets.ln_prob(Some(&(0, 0))), Some(std::f64::NEG_INFINITY));
    }

    #[test]
    fn child_bucket_sanity_test() {
        let mut buckets = Categorical::new();
        buckets.add_child_pop(Some((0, 0)), 5.0);
        assert_approx_eq!(buckets.ln_prob(Some(&(0, 0))).unwrap(), 0.0);
        assert_approx_eq!(buckets.kl_divergence(&buckets).unwrap(), 0.0);
        assert_eq!(buckets.ln_prob(None).unwrap(), std::f64::NEG_INFINITY);
    }

    #[test]
    fn mixed_bucket_sanity_test() {
        let mut bucket1 = Categorical::new();
        bucket1.add_child_pop(None, 6.0);
        bucket1.add_child_pop(Some((0, 0)), 6.0);
        println!("{:?}", bucket1);

        let mut bucket2 = Categorical::new();
        bucket2.add_child_pop(None, 4.0);
        bucket2.add_child_pop(Some((0, 0)), 8.0);
        println!("{:?}", bucket2);

        assert_approx_eq!(bucket1.ln_prob(None).unwrap(), (0.5f64).ln());
        assert_approx_eq!(
            bucket2.ln_prob(Some(&(0, 0))).unwrap(),
            (0.666666666f64).ln()
        );
        assert_approx_eq!(bucket1.kl_divergence(&bucket1).unwrap(), 0.0);

        assert_approx_eq!(bucket1.kl_divergence(&bucket2).unwrap(), 0.05889151782);
        assert_approx_eq!(bucket2.kl_divergence(&bucket1).unwrap(), 0.05663301226);
    }
}
