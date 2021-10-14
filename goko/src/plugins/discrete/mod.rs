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

pub mod categorical;
pub mod dirichlet;
pub mod tracker;

#[allow(unused_imports)]
pub mod prelude {
    //! Easy way of importing everything
    pub use super::categorical::*;
    pub use super::dirichlet::*;
    pub use super::tracker::*;
}
