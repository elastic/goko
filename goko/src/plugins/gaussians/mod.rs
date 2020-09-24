//! # Probability Distributions Plugins
//!
//! This module containes plugins that simulate probability distributions on the nodes.
//! It also has trackers used to see when queries and sequences are out of distribution.

use super::*;
use rand::Rng;
use std::fmt::Debug;

mod diag_gaussian;
pub use diag_gaussian::*;

mod svd_gaussian;
pub use svd_gaussian::*;

///
pub trait ContinousDistribution: Clone + 'static {
    /// Pass none if you want to test for a singleton, returns 0 if
    fn ln_pdf(&self, point: &PointRef) -> Option<f64>;
    /// Samples a point from this distribution
    fn sample<R: Rng>(&self, rng: &mut R) -> Vec<f32>;

    /// Computes the KL divergence of two bucket probs.
    /// KL(self || other)
    /// Returns None if the support of the self is not a subset of the support of the other, or the calculation is undefined.
    fn kl_divergence(&self, other: &Self) -> Option<f64>;
}

///
pub trait ContinousBayesianDistribution: ContinousDistribution + Clone + 'static {
    /// Adds an observation to the distribution.
    /// This currently shifts the underlying parameters of the distribution rather than be tracked.
    fn add_observation(&mut self, point: &PointRef);
}