use pointcloud::*;

use goko::PartitionType;
use serde::{Deserialize, Serialize};
use crate::core::*;
use goko::errors::GokoError;

/// Send a `GET` request to `/` for this
#[derive(Deserialize, Serialize, Clone, Copy)]
pub struct ParametersRequest;

/// Response to a parameters request
#[derive(Deserialize, Serialize)]
pub struct ParametersResponse {
    /// See paper or main description, governs the number of children of each node. Higher is more.
    pub scale_base: f32,
    /// If a node covers less than or equal to this number of points, it becomes a leaf.
    pub leaf_cutoff: usize,
    /// If a node has scale index less than or equal to this, it becomes a leaf
    pub min_res_index: i32,
    /// If you don't want singletons messing with your tree and want everything to be a node or a element of leaf node, make this true.
    pub use_singletons: bool,
    /// The partition type of the tree
    pub partition_type: PartitionType,
    /// This should be replaced by a logging solution
    pub verbosity: u32,
    /// The seed to use for deterministic trees. This is xor-ed with the point index to create a seed for `rand::rngs::SmallRng`.
    pub rng_seed: Option<u64>,
}

impl ParametersRequest {
    pub fn process<D: PointCloud>(self, reader: &mut CoreReader<D>) -> Result<ParametersResponse, GokoError> {
        let params = reader.tree.parameters();
        Ok(ParametersResponse {
            scale_base: params.scale_base,
            leaf_cutoff: params.leaf_cutoff,
            min_res_index: params.min_res_index,
            use_singletons: params.use_singletons,
            partition_type: params.partition_type,
            verbosity: params.verbosity,
            rng_seed: params.rng_seed,
        })
    }
}
