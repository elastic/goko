//! # Dirichlet probability
//!
//! We know that the users are quering based on what they want to know about.
//! This has some geometric structure, especially for attackers. We have some
//! prior knowlege about the queries, they should function similarly to the training set.
//! Sections of data that are highly populated should have a higher likelyhood of being
//! queried.
//!
//! This plugin lets us simulate the unknown distribution of the queries of a user in a
//! bayesian way. There may be more applications of this idea, but defending against
//! attackers has been proven.

use crate::covertree::node::CoverNode;
use crate::covertree::CoverTreeReader;
use crate::plugins::*;

use rand::prelude::*;
use statrs::function::gamma::{digamma, ln_gamma};

use rand::distributions::{Distribution, Uniform};

use super::categorical::*;

/// Simple probability density function for where things go by count
///

impl<D: PointCloud> NodePlugin<D> for Dirichlet {}

/// Stores the log probabilities for each node in the tree.
///
/// This is the probability that when you sample from the tree you end up at a particular node.
#[derive(Debug, Clone, Default)]
pub struct GokoDirichlet {
    // probability that you'd pass thru this node.
//pub cond_ln_probs: HashMap<NodeAddress,f64>,
}

/// Parent trait that make this all work. Ideally this should be included in the `TreePlugin` but rust doesn't like it.
impl<D: PointCloud> GokoPlugin<D> for GokoDirichlet {
    type NodeComponent = Dirichlet;
    fn node_component(
        _parameters: &Self,
        my_node: &CoverNode<D>,
        my_tree: &CoverTreeReader<D>,
    ) -> Option<Self::NodeComponent> {
        let mut bucket = Dirichlet::new();

        // If we're a routing node then grab the childen's values
        if let Some(child_addresses) = my_node.children() {
            for ca in child_addresses {
                my_tree.get_node_plugin_and::<Self::NodeComponent, _, _>(*ca, |p| {
                    bucket.add_child_pop(Some(*ca), p.total());
                });
            }
            bucket.add_child_pop(NodeAddress::, my_node.singletons_len() as f64);
        } else {
            bucket.add_child_pop(None, (my_node.singletons_len() + 1) as f64);
        }
        Some(bucket)
    }

    /*
    fn tree_component(parameters: &mut Self, my_tree: &mut CoverTreeWriter<D>) {
        let mut unvisited_nodes = vec![my_tree.root_address];
        let reader = my_tree.reader();
        parameters.cond_ln_probs.insert(my_tree.root_address, 0.0);
        while let Some(addr) = unvisited_nodes.pop() {
            let pass_thru_prob = parameters.cond_ln_probs.get(&addr).unwrap().clone();
            let ln_probs = reader.get_node_plugin_and::<Self::NodeComponent,_,_>(addr, |p| p.ln_prob_vector()).unwrap();
            if let Some((child_probs,_singleton_prob)) = ln_probs {
                for (child_addr,child_prob) in child_probs {
                    parameters.cond_ln_probs.insert(child_addr, pass_thru_prob + child_prob);
                    unvisited_nodes.push(child_addr);
                }
            }
        }
    }
    */
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    //use crate::covertree::tests::build_basic_tree;

    #[test]
    fn dirichlet_sanity_test() {
        let mut buckets = Dirichlet::new();
        buckets.add_child_pop(None, 5.0);
        println!("{:?}", buckets);
        assert_approx_eq!(buckets.ln_pdf(None).unwrap(), 0.0);
        assert_approx_eq!(buckets.kl_divergence(&buckets).unwrap(), 0.0);
    }

    #[test]
    fn dirichlet_mixed_sanity_test() {
        let mut buckets = Dirichlet::new();
        buckets.add_child_pop(None, 5.0);
        buckets.add_child_pop(Some((0, 0).into()), 5.0);
        println!("{:?}", buckets);
        assert_approx_eq!(buckets.ln_pdf(None).unwrap(), 0.5f64.ln());
        assert_approx_eq!(buckets.ln_pdf(Some(&(0, 0).into())).unwrap(), 0.5f64.ln());
        assert_approx_eq!(buckets.kl_divergence(&buckets).unwrap(), 0.0);
    }

    #[test]
    fn dirichlet_posterior_sanity_test() {
        let mut buckets = Dirichlet::new();
        buckets.add_child_pop(None, 3.0);
        buckets.add_child_pop(Some((0, 0).into()), 5.0);

        let mut categorical = Categorical::new();
        categorical.add_child_pop(None, 2.0);

        let mut buckets_posterior = buckets.clone();
        buckets_posterior.add_evidence(&categorical);

        println!("Buckets: {:?}", buckets);
        println!("Buckets Posterior: {:?}", buckets_posterior);
        println!("Evidence: {:?}", categorical);
        assert_approx_eq!(buckets_posterior.ln_pdf(None).unwrap(), 0.5f64.ln());
        assert_approx_eq!(
            buckets_posterior.ln_pdf(Some(&(0, 0).into())).unwrap(),
            0.5f64.ln()
        );
        assert_approx_eq!(
            buckets.kl_divergence(&buckets_posterior).unwrap(),
            buckets.posterior_kl_divergence(&categorical).unwrap()
        );
    }

    #[test]
    fn dirichlet_posterior_sanity_test_2() {
        let mut buckets = Dirichlet::new();
        buckets.add_child_pop(None, 3.0);
        buckets.add_child_pop(Some((0, 0).into()), 2.0);

        let mut categorical = Categorical::new();
        categorical.add_child_pop(None, 3.0);
        categorical.add_child_pop(Some((0, 0).into()), 2.0);

        let mut buckets_posterior = buckets.clone();
        buckets_posterior.add_evidence(&categorical);

        println!("Buckets: {:?}", buckets);
        println!("Buckets Posterior: {:?}", buckets_posterior);
        println!("Evidence: {:?}", categorical);
        assert_approx_eq!(
            buckets.kl_divergence(&buckets_posterior).unwrap(),
            buckets.posterior_kl_divergence(&categorical).unwrap()
        );
        assert_approx_eq!(
            buckets.kl_divergence(&buckets_posterior).unwrap(),
            0.1789970483832892
        );
    }

    #[test]
    fn dirichlet_posterior_sanity_test_3() {
        let mut buckets = Dirichlet::new();
        buckets.add_child_pop(None, 3.0);

        let mut categorical = Categorical::new();
        categorical.add_child_pop(None, 3.0);

        let mut buckets_posterior = buckets.clone();
        buckets_posterior.add_evidence(&categorical);

        println!("Buckets: {:?}", buckets);
        println!("Buckets Posterior: {:?}", buckets_posterior);
        println!("Evidence: {:?}", categorical);
        assert_approx_eq!(
            buckets.kl_divergence(&buckets_posterior).unwrap(),
            buckets.posterior_kl_divergence(&categorical).unwrap()
        );
        assert_approx_eq!(buckets.kl_divergence(&buckets_posterior).unwrap(), 0.0);
    }

    #[test]
    fn dirichlet_kl_sanity_test() {
        let mut bucket1 = Dirichlet::new();
        bucket1.add_child_pop(None, 6.0);
        bucket1.add_child_pop(Some((0, 0).into()), 6.0);
        println!("{:?}", bucket1);

        let mut bucket2 = Dirichlet::new();
        bucket2.add_child_pop(None, 3.0);
        bucket2.add_child_pop(Some((0, 0).into()), 9.0);
        println!("{:?}", bucket2);

        let mut bucket3 = Dirichlet::new();
        bucket3.add_child_pop(None, 5.5);
        bucket3.add_child_pop(Some((0, 0).into()), 6.5);
        println!("{:?}", bucket3);
        println!(
            "{:?}, {}",
            bucket1.kl_divergence(&bucket2).unwrap(),
            bucket1.kl_divergence(&bucket3).unwrap()
        );
        assert!(
            bucket1.kl_divergence(&bucket2).unwrap() > bucket1.kl_divergence(&bucket3).unwrap()
        );
    }

    /*
    #[test]
    fn dirichlet_tree_probs_test() {
        let mut tree = build_basic_tree();
        tree.add_plugin::<GokoDirichlet>(GokoDirichlet::default());
        assert!(tree.reader().get_plugin_and::<GokoDirichlet,_,_>(|p| {
            // Checked these probs by hand
            assert_approx_eq!(p.cond_ln_probs.get(&( -2,2)).unwrap(),-0.5108256237659905);
            true
        }).unwrap());
    }
    */
}
