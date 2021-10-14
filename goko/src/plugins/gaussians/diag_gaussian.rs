//! # Diagonal Gaussian
//!
//! This computes a coordinate bound multivariate Gaussian. This can be thought of as a rough
//! simulation of the data underling a node. However we can chose the scale from which we
//! simulate the data, down to the individual point, so this can be arbitrarily accurate.

use super::*;
use crate::covertree::node::CoverNode;
use crate::covertree::CoverTreeReader;

use rand::prelude::*;
use rand_distr::StandardNormal;
use std::f32::consts::PI;

/// Node component, coded in such a way that it can be efficiently, recursively computed.
#[derive(Debug, Clone, Default)]
pub struct DiagGaussian {
    /// First Moment
    pub moment1: Vec<f32>,
    /// Second Moment
    pub moment2: Vec<f32>,
    /// Cover count, divide the first moment by this to get the mean.
    pub count: usize,
}

macro_rules! internal_mean {
    ( $m1:expr,$c:expr ) => {{
        $m1.iter().map(|x| x / ($c as f32))
    }};
}
macro_rules! internal_var {
    ( $m1:expr,$m2:expr,$c:expr ) => {{
        $m2.iter()
            .map(|x| x / ($c as f32))
            .zip($m1.iter().map(|x| x / ($c as f32)))
            .map(|(x, u)| x - u * u)
    }};
}

impl ContinousDistribution for DiagGaussian {
    fn ln_pdf<T: PointRef>(&self, point: &T) -> Option<f64> {
        let mean_vars = internal_mean!(self.moment1, self.count).zip(internal_var!(
            self.moment1,
            self.moment2,
            self.count
        ));

        let (exponent, det) = point
            .dense_iter()
            .zip(mean_vars)
            .map(|(xi, (ui, vi))| ((xi - ui) * (xi - ui) / vi, vi))
            .fold((0.0, 1.0), |(a, v), (x, u)| (a + x, v * u));

        Some((exponent - det + (self.dim() as f32) * (2.0 * PI).ln()) as f64)
    }

    fn sample<R: Rng>(&self, rng: &mut R) -> Vec<f32> {
        let mean_iter = internal_mean!(self.moment1, self.count);
        let std_dev_iter = internal_var!(self.moment1, self.moment2, self.count).map(|f| f.sqrt());

        StandardNormal
            .sample_iter(rng)
            .take(self.moment1.len())
            .zip(mean_iter.zip(std_dev_iter))
            .map(|(s, (a, b)): (f32, _)| s * b + a)
            .collect()
    }

    fn kl_divergence(&self, other: &DiagGaussian) -> Option<f64> {
        let mean_vars = internal_mean!(self.moment1, self.count).zip(internal_var!(
            self.moment1,
            self.moment2,
            self.count
        ));

        let other_mean_vars = internal_mean!(other.moment1, other.count).zip(internal_var!(
            other.moment1,
            other.moment2,
            other.count
        ));

        let (trace, mah_dist, ln_det) = mean_vars
            .zip(other_mean_vars)
            .map(|((xi, yi), (ui, vi))| {
                let trace = yi / ui;
                let mah_dist = (ui - xi) * (ui - xi) / vi;
                let ln_det = vi.ln() - yi.ln();
                (trace, mah_dist, ln_det)
            })
            .fold((0.0, 0.0, 0.0), |(a, b, c), (x, y, z)| {
                (a + x, b + y, c + z)
            });

        Some(((trace + mah_dist - (self.moment1.len() as f32) + ln_det) / 2.0) as f64)
    }
}

impl DiagGaussian {
    /// Creates a new empty diagonal gaussian
    pub fn new(dim: usize) -> DiagGaussian {
        DiagGaussian {
            moment1: vec![0.0; dim],
            moment2: vec![0.0; dim],
            count: 0,
        }
    }

    /// Dimension for this
    pub fn dim(&self) -> usize {
        self.moment1.len()
    }

    /// adds a point to the Diagonal Gaussian
    pub fn add_point<T: PointRef>(&mut self, point: &T) {
        self.moment1
            .iter_mut()
            .zip(point.dense_iter())
            .for_each(|(m, p)| *m += p);
        self.moment2
            .iter_mut()
            .zip(point.dense_iter())
            .for_each(|(m, p)| *m += p * p);
        self.count += 1;
    }

    /// removes a point from the Diagonal Gaussian
    pub fn remove_point<T: PointRef>(&mut self, point: &T) {
        if self.count != 0 {
            self.moment1
                .iter_mut()
                .zip(point.dense_iter())
                .for_each(|(m, p)| *m -= p);
            self.moment2
                .iter_mut()
                .zip(point.dense_iter())
                .for_each(|(m, p)| *m -= p * p);
            self.count += 1;
        }
    }

    /// Merges two diagonal gaussians together
    pub fn merge(&mut self, other: &DiagGaussian) {
        self.moment1
            .iter_mut()
            .zip(other.moment1.iter())
            .for_each(|(m, p)| *m += *p);
        self.moment2
            .iter_mut()
            .zip(other.moment2.iter())
            .for_each(|(m, p)| *m += *p);
        self.count += other.count;
    }

    /// Mean: `moment1/count`
    pub fn mean(&self) -> Vec<f32> {
        if self.count > 0 {
            internal_mean!(self.moment1, self.count).collect()
        } else {
            vec![0.0; self.moment1.len()]
        }
    }
    /// Variance: `moment2/count - (moment1/count)^2`
    pub fn var(&self) -> Vec<f32> {
        if self.count > 0 {
            internal_var!(self.moment1, self.moment2, self.count).collect()
        } else {
            vec![0.0; self.moment1.len()]
        }
    }
    /// The count used in the above equations
    pub fn count(&self) -> usize {
        self.count
    }
}

impl<D: PointCloud> NodePlugin<D> for DiagGaussian {}

/// Zero sized type that can be passed around. Equivilant to `()`
#[derive(Debug, Clone)]
pub struct GokoDiagGaussian {
    recursive: bool,
}

impl GokoDiagGaussian {
    /// Sets this up to build the gaussians recursively, so the gaussian for a node is for the total cover space.
    pub fn recursive() -> Self {
        Self { recursive: true }
    }

    /// Produces a gaussian off of just the singletons attached to the node, not the total cover space
    pub fn singletons() -> Self {
        Self { recursive: false }
    }
}

impl<D: PointCloud> GokoPlugin<D> for GokoDiagGaussian {
    type NodeComponent = DiagGaussian;
    fn node_component(
        parameters: &Self,
        my_node: &CoverNode<D>,
        my_tree: &CoverTreeReader<D>,
    ) -> Option<Self::NodeComponent> {
        let moment1 = my_tree
            .parameters()
            .point_cloud
            .moment_1(my_node.singletons())
            .unwrap();
        let moment2 = my_tree
            .parameters()
            .point_cloud
            .moment_2(my_node.singletons())
            .unwrap();
        let count = my_node.singletons_len();
        let mut my_dg = DiagGaussian {
            moment1,
            moment2,
            count,
        };
        // If we're a routing node then grab the childen's values
        if let Some(child_addresses) = my_node.children() {
            if parameters.recursive {
                for ca in child_addresses {
                    my_tree.get_node_plugin_and::<Self::NodeComponent, _, _>(*ca, |p| {
                        my_dg.merge(p);
                    });
                }
            }
        } else {
            my_dg.add_point(
                &my_tree
                    .parameters()
                    .point_cloud
                    .point(my_node.center_index())
                    .unwrap(),
            );
        }
        Some(my_dg)
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::covertree::tests::build_basic_tree;

    #[test]
    fn recursive_gaussian_sanity_correct() {
        let basic_tree_data = vec![0.499, 0.49, 0.48, -0.49, 0.0];
        let moment1 = basic_tree_data.iter().fold(0.0, |a, x| a + x);
        let moment2 = basic_tree_data.iter().fold(0.0, |a, x| a + x * x);
        let count = basic_tree_data.len();
        let mut tree = build_basic_tree();
        tree.add_plugin::<GokoDiagGaussian>(GokoDiagGaussian::recursive());
        println!("{:?}", tree.reader().len());
        let reader = tree.reader();

        for (si, layer) in tree.reader().layers() {
            println!("Scale Index: {:?}", si);
            layer.for_each_node(|_pi, n| {
                if n.is_leaf() {
                    n.get_plugin_and::<DiagGaussian, _, _>(|p| {
                        println!(
                            "=====<Leaf ({},{})>=====",
                            n.scale_index(),
                            n.center_index()
                        );
                        println!(
                            "DiagGauss: {:?}, Singles: {:?}, Center Index: {:?}",
                            p,
                            n.singletons(),
                            n.center_index()
                        );
                        let singles = n.singletons();
                        let mut s_moment1 = singles
                            .iter()
                            .map(|i| basic_tree_data[*i as usize])
                            .fold(0.0, |a, x| a + x);
                        let mut s_moment2 = singles
                            .iter()
                            .map(|i| basic_tree_data[*i as usize])
                            .fold(0.0, |a, x| a + x * x);
                        s_moment1 += basic_tree_data[*n.center_index() as usize];
                        s_moment2 += basic_tree_data[*n.center_index() as usize]
                            * basic_tree_data[*n.center_index() as usize];
                        let s_count = singles.len() + 1;
                        println!(
                            "First moment, expected: {:?}, calculated: {:?}",
                            s_moment1, p.moment1[0]
                        );
                        assert_approx_eq!(s_moment1, p.moment1[0]);
                        println!(
                            "Second moment, expected: {:?}, calculated: {:?}",
                            s_moment2, p.moment2[0]
                        );
                        assert_approx_eq!(s_moment2, p.moment2[0]);
                        assert_eq!(s_count, p.count);
                        println!(
                            "=====</Leaf ({},{})>=====",
                            n.scale_index(),
                            n.center_index()
                        );
                    });
                } else {
                    n.get_plugin_and::<DiagGaussian, _, _>(|dp| {
                        println!(
                            "=====<Routing ({},{})>=====",
                            n.scale_index(),
                            n.center_index()
                        );
                        println!("DiagGauss: {:?}", dp);
                        println!(
                            "=====</Routing ({},{})>=====",
                            n.scale_index(),
                            n.center_index()
                        );
                    });
                }
            });
        }

        reader.get_node_plugin_and::<DiagGaussian, _, _>(reader.root_address(), |p| {
            println!(
                "First moment, expected: {:?}, calculated: {:?}",
                moment1, p.moment1[0]
            );
            assert_approx_eq!(moment1, p.moment1[0]);
            println!(
                "Second moment, expected: {:?}, calculated: {:?}",
                moment2, p.moment2[0]
            );
            assert_approx_eq!(moment2, p.moment2[0]);
            assert_eq!(count, p.count);
        });
    }

    #[test]
    fn diag_gaussian_sanity_check() {
        let mut ct = build_basic_tree();
        ct.add_plugin::<GokoDiagGaussian>(GokoDiagGaussian::recursive());
        let ct_reader = ct.reader();
        let mut untested_addresses = vec![ct_reader.root_address()];
        while let Some(addr) = untested_addresses.pop() {
            let count = ct_reader
                .get_node_plugin_and::<DiagGaussian, _, _>(addr, |p| {
                    p.mean()
                        .iter()
                        .for_each(|f| assert!(f.is_finite(), "Mean: {}, at address {:?}", f, addr));
                    p.var().iter().for_each(|f| {
                        assert!(f.is_finite(), "Variance: {}, at address {:?}", f, addr)
                    });
                    p.count
                })
                .unwrap();
            ct_reader.get_node_and(addr, |n| {
                assert_eq!(n.coverage_count(), count, "Node: {:?}", n)
            });

            ct_reader.get_node_children_and(addr, |covered, children| {
                untested_addresses.push(covered);
                untested_addresses.extend(children);
            });
        }
    }
}
