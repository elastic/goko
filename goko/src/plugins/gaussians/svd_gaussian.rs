//! # Diagonal Gaussian
//!
//! This computes a coordinate bound multivariate Gaussian. This can be thought of as a rough
//! simulation of the data underling a node. However we can chose the scale from which we
//! simulate the data, down to the individual point, so this can be arbitrarily accurate.

use super::*;
use crate::covertree::node::CoverNode;
use crate::covertree::CoverTreeReader;
use crate::plugins::utils::*;

use ndarray::prelude::*;
use ndarray_linalg::svd::*;

/// Node component, coded in such a way that it can be efficiently, recursively computed.
#[derive(Debug, Clone, Default)]
pub struct SvdGaussian {
    /// Mean of this gaussian
    pub mean: Array1<f32>,
    /// Second Moment
    pub vt: Array2<f32>,
    /// The singular values
    pub singular_vals: Array1<f32>,
}
/*
impl ContinousDistribution for SvdGaussian {
    fn ln_pdf(&self, _point: &PointRef) -> Option<f64> {
        unimplemented!()
    }

    fn sample<R: Rng>(&self, _rng: &mut R) -> Vec<f32> {
        unimplemented!()
    }

    fn kl_divergence(&self, _other: &SvdGaussian) -> Option<f64> {
        unimplemented!()
    }
}
*/
impl SvdGaussian {
    /// Mean: `moment1/count`
    pub fn mean(&self) -> Array1<f32> {
        self.mean.clone()
    }
}

impl<D: PointCloud> NodePlugin<D> for SvdGaussian {}

/// Zero sized type that can be passed around. Equivilant to `()`
#[derive(Debug, Clone)]
pub struct GokoSvdGaussian {
    max_points: usize,
    min_points: usize,
    tau: f32,
}

impl GokoSvdGaussian {
    /// Specify the max number of points, and the min that you want to compute the SVD over, and the tau used for dimension calulations
    pub fn new(min_points: usize, max_points: usize, tau: f32) -> GokoSvdGaussian {
        GokoSvdGaussian {
            max_points,
            min_points,
            tau,
        }
    }
}

impl<D: PointCloud> GokoPlugin<D> for GokoSvdGaussian {
    type NodeComponent = SvdGaussian;
    fn prepare_tree(parameters: &Self, my_tree: &mut CoverTreeWriter<D>) {
        my_tree.add_plugin::<GokoCoverageIndexes>(GokoCoverageIndexes::restricted(
            parameters.max_points,
        ));
        my_tree.add_plugin::<GokoDiagGaussian>(GokoDiagGaussian::recursive());
    }
    fn node_component(
        parameters: &Self,
        my_node: &CoverNode<D>,
        my_tree: &CoverTreeReader<D>,
    ) -> Option<Self::NodeComponent> {
        if my_node.coverage_count() > parameters.min_points {
            let points = my_node.get_plugin_and::<CoverageIndexes, _, _>(|p| {
                my_tree
                    .parameters()
                    .point_cloud
                    .points_dense_matrix(p.point_indexes())
                    .unwrap()
            });
            if let Some(mut points) = points {
                let mean = my_node
                    .get_plugin_and::<DiagGaussian, _, _>(|p| {
                        Array1::from_shape_vec((p.dim(),), p.mean()).unwrap()
                    })
                    .unwrap();
                for mut p in points.axis_iter_mut(Axis(0)) {
                    p -= &mean;
                }

                let (_u, singular_vals, vt) = points.svd(false, true).unwrap();
                let vt = vt.unwrap();
                Some(SvdGaussian {
                    singular_vals,
                    vt,
                    mean,
                })
            } else {
                None
            }
        } else {
            None
        }
    }
}
/*
#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::covertree::tests::build_basic_tree;
}
*/
