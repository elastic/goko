#[cfg(test)]
#[macro_use]
extern crate assert_approx_eq;

use std::convert::From;

mod categorical;
mod dirichlet;
mod parameter_store;
mod stats_consts;
pub use categorical::Categorical;
pub use dirichlet::{Dirichlet, DirichletTracker};
