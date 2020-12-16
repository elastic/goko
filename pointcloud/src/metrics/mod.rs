//! Metrics. 


// Todo: Make this a macro
mod l2_misc;
pub use l2_misc::*;
mod l1_misc;
pub use l1_misc::*;
mod l2_f32;
pub use l2_f32::*;
mod l1_f32;
pub use l1_f32::*;

#[derive(Debug)]
/// L2 distance trait.
pub struct L2 {}
/// L1 distance trait
pub struct L1 {}

