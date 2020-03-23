//! # Diagonal Gaussian
//!
//! This computes a coordinate bound multivariate Gaussian.

use super::*;
use std::f32::consts::{E,PI};


/// Node component, coded in such a way that it can be efficiently, recursively computed.
#[derive(Debug, Clone)]
pub struct DiagGaussian {
    /// First Moment
    pub moment1: Vec<f32>,
    /// Second Moment
    pub moment2: Vec<f32>,
    /// Cover count, divide the first moment by this to get the mean.
    pub count: usize,
}

impl DiagGaussian {
    /// Creates a new empty diagonal gaussian
    pub fn new() -> DiagGaussian {
        DiagGaussian { 
            moment1: Vec::new(),
            moment2: Vec::new(),
            count: 0,
        }
    }

    /// adds a point to the Diagonal Gaussian
    pub fn add_point(&mut self, point:&[f32]) {
        if self.count == 0 {
            self.moment1.extend(point);
            self.moment2.extend(point.iter().map(|v|v*v));
            self.count = 1;
        } else {
            self.moment1.iter_mut().zip(point).for_each(|(m,p)| *m += p );
            self.moment2.iter_mut().zip(point).for_each(|(m,p)| *m += p*p );
            self.count += 1;
        }
    }

    /// removes a point from the Diagonal Gaussian
    pub fn remove_point(&mut self, point:&[f32]) {
        if self.count != 0 {
            self.moment1.iter_mut().zip(point).for_each(|(m,p)| *m -= p );
            self.moment2.iter_mut().zip(point).for_each(|(m,p)| *m -= p*p );
            self.count += 1;
        }
    }

    /// Merges two diagonal gaussians together
    pub fn merge(&mut self, other: &DiagGaussian) {
        if self.count == 0 {
            self.moment1 = other.moment1.clone();
            self.moment2 = other.moment2.clone();
            self.count = other.count;
        } else {
            self.moment1.iter_mut().zip(other.moment1.iter()).for_each(|(m,p)| *m += *p );
            self.moment2.iter_mut().zip(other.moment2.iter()).for_each(|(m,p)| *m += *p );
            self.count += other.count;
        }
    }

    /// Mean: `moment1/count`
    pub fn mean(&self) -> Vec<f32> {
        self.moment1.iter().map(|x| x / (self.count as f32)).collect()
    }
    /// Variance: `moment2/count - (moment1/count)^2`
    pub fn var(&self) -> Vec<f32> {
        self.moment2
            .iter()
            .map(|x| x / (self.count as f32))
            .zip(self.moment1.iter().map(|x| x / (self.count as f32)))
            .map(|(x, u)| x - u * u)
            .collect()
    }

    /// Computes the probability density function of the gaussian
    pub fn pdf(&self, point:&[f32]) -> f32 {
        let means = self.moment1.iter().map(|x| x / (self.count as f32));
        let vars = self.moment2
            .iter()
            .map(|x| x / (self.count as f32))
            .zip(self.moment1.iter().map(|x| x / (self.count as f32)))
            .map(|(x, u)| x - u * u);
        let mean_vars = means.zip(vars);

        let (exponent,det) = point.iter().zip(mean_vars).map(|(xi,(ui,vi))| {
            ((xi-ui)*(xi-ui)/vi,vi)
        }).fold((0.0,1.0),|(a,v),(x,u)| (a+x,v*u));

        E.powf(exponent)/(det*(2.0*PI).powi(point.len() as i32))
    }

    /// Measures the divergence between this and another gaussian
    pub fn kl_divergence(&self, other:&DiagGaussian) -> f32 {
        let means = self.moment1.iter().map(|x| x / (self.count as f32));
        let vars = self.moment2
            .iter()
            .map(|x| x / (self.count as f32))
            .zip(self.moment1.iter().map(|x| x / (self.count as f32)))
            .map(|(x, u)| x - u * u);
        let mean_vars = means.zip(vars);

        let other_means = other.moment1.iter().map(|x| x / (other.count as f32));
        let other_vars = other.moment2
            .iter()
            .map(|x| x / (other.count as f32))
            .zip(other.moment1.iter().map(|x| x / (other.count as f32)))
            .map(|(x, u)| x - u * u);
        let other_mean_vars = other_means.zip(other_vars);

        let (trace,mah_dist,det,other_det) = mean_vars.zip(other_mean_vars).map(|((xi,yi),(ui,vi))| {
            let trace = yi/ui;
            let mah_dist = (ui-xi)*(ui-xi)/vi;
            let det = yi;
            let other_det = vi;
            (trace,mah_dist,det,other_det)
        }).fold((0.0,0.0,1.0f32,1.0f32),|(a,b,c,d),(x,y,z,w)| (a+x,b+y,c*z,d*w));

        (trace + mah_dist - (self.moment1.len() as f32) + (other_det/det).ln())/2.0
    }
}

impl<M: Metric> NodePlugin<M> for DiagGaussian {
    fn update(&mut self, _my_node: &CoverNode<M>, _my_tree: &CoverTreeReader<M>) {}
}

/// Zero sized type that can be passed around. Equivilant to `()`
#[derive(Debug, Clone)]
pub struct DiagGaussianTree {
    recursive: bool,
}

impl<M: Metric> TreePlugin<M> for DiagGaussianTree {
    fn update(&mut self, _my_tree: &CoverTreeReader<M>) {}
}

/// Zero sized type that can be passed around. Equivilant to `()`
pub struct GrandmaDiagGaussian {}

impl GrandmaDiagGaussian {
    /// Sets this up to build the gaussians recursively, so the gaussian for a node is for the total cover space.
    pub fn recursive() -> DiagGaussianTree {
        DiagGaussianTree {
            recursive: true
        }
    }

    /// Produces a gaussian off of just the singletons attached to the node, not the total cover space
    pub fn singletons() -> DiagGaussianTree {
        DiagGaussianTree {
            recursive: false
        }
    }
}

impl<M: Metric> GrandmaPlugin<M> for GrandmaDiagGaussian {
    type NodeComponent = DiagGaussian;
    type TreeComponent = DiagGaussianTree;
    fn node_component(
        parameters: &Self::TreeComponent,
        my_node: &CoverNode<M>,
        my_tree: &CoverTreeReader<M>,
    ) -> Self::NodeComponent {
        let moment1 = my_tree
            .parameters()
            .point_cloud
            .moment_subset(1, my_node.singletons())
            .unwrap();
        let moment2 = my_tree
            .parameters()
            .point_cloud
            .moment_subset(2, my_node.singletons())
            .unwrap();
        let count = my_node.singleton_len();
        let mut my_dg = DiagGaussian { moment1, moment2, count };

        // If we're a routing node then grab the childen's values
        if let Some((nested_scale, child_addresses)) = my_node.children() {
            if parameters.recursive {
                my_tree.get_node_plugin_and::<Self::NodeComponent, _, _>(
                    (nested_scale, *my_node.center_index()),
                    |p| {
                        my_dg.merge(p);
                    },
                );
                for ca in child_addresses {
                    my_tree.get_node_plugin_and::<Self::NodeComponent, _, _>(*ca, |p| {
                        my_dg.merge(p);
                    });
                }
            }
        } else {
            my_dg.add_point(my_tree
                .parameters()
                .point_cloud
                .get_point(*my_node.center_index())
                .unwrap());
        }
        my_dg
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tree::tests::build_basic_tree;

    #[test]
    fn recursive_gaussian_sanity_correct() {
        let basic_tree_data = vec![0.499, 0.49, 0.48, -0.49, 0.0];
        let moment1 = basic_tree_data.iter().fold(0.0, |a, x| a + x);
        let moment2 = basic_tree_data.iter().fold(0.0, |a, x| a + x * x);
        let count = basic_tree_data.len();
        let mut tree = build_basic_tree();
        tree.add_plugin::<GrandmaDiagGaussian>(GrandmaDiagGaussian::recursive());
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
}
