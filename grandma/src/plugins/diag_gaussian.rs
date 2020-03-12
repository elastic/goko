//! # Diagonal Gaussian
//! 
//! This computes a coordinate bound multivariate Gaussian. 

use super::*;

/// Node component, coded in such a way that it can be efficiently, recursively computed.
#[derive(Debug, Clone)]
pub struct DiagGaussianNode {
    /// First Moment
    pub mom1: Vec<f32>,
    /// Second Moment
    pub mom2: Vec<f32>,
    /// Cover count, divide the first moment by this to get the mean.
    pub count: usize,
}

impl DiagGaussianNode {
    /// Mean: `mom1/count`
    pub fn mean(&self) -> Vec<f32> {
        self.mom1.iter().map(|x| x / (self.count as f32)).collect()
    }
    /// Variance: `mom2/count - (mom1/count)^2`
    pub fn var(&self) -> Vec<f32> {
        self.mom2
            .iter()
            .map(|x| x / (self.count as f32))
            .zip(self.mom1.iter().map(|x| x / (self.count as f32)))
            .map(|(x, u)| x - u * u)
            .collect()
    }
}

impl<M: Metric> NodePlugin<M> for DiagGaussianNode {
    fn update(&mut self, _my_node: &CoverNode<M>, _my_tree: &CoverTreeReader<M>) {}
}

/// Zero sized type that can be passed around. Equivilant to `()`
#[derive(Debug, Clone)]
pub struct DiagGaussianTree {}

impl<M: Metric> TreePlugin<M> for DiagGaussianTree {
    fn update(&mut self, _my_tree: &CoverTreeReader<M>) {}
}

/// Zero sized type that can be passed around. Equivilant to `()`
pub struct GrandmaDiagGaussian {}   

impl<M: Metric> GrandmaPlugin<M> for GrandmaDiagGaussian {
    type NodeComponent = DiagGaussianNode;
    type TreeComponent = DiagGaussianTree;
    fn node_component(
        _parameters: &Self::TreeComponent,
        my_node: &CoverNode<M>,
        my_tree: &CoverTreeReader<M>,
    ) -> Self::NodeComponent {
        let mut mom1 = my_tree
            .parameters()
            .point_cloud
            .moment_subset(1, my_node.singletons())
            .unwrap();
        let mut mom2 = my_tree
            .parameters()
            .point_cloud
            .moment_subset(2, my_node.singletons())
            .unwrap();
        let mut count = my_node.singleton_len();

        // If we're a routing node then grab the childen's values
        if let Some((nested_scale, child_addresses)) = my_node.children() {
            my_tree.get_node_plugin_and::<Self::NodeComponent, _, _>(
                (nested_scale, *my_node.center_index()),
                |p| {
                    for (m, yy) in mom1.iter_mut().zip(&p.mom1) {
                        *m += yy;
                    }
                    for (m, yy) in mom2.iter_mut().zip(&p.mom2) {
                        *m += yy;
                    }
                    count += p.count;
                },
            );
            for ca in child_addresses {
                my_tree.get_node_plugin_and::<Self::NodeComponent, _, _>(*ca, |p| {
                    for (m, yy) in mom1.iter_mut().zip(&p.mom1) {
                        *m += yy;
                    }
                    for (m, yy) in mom2.iter_mut().zip(&p.mom2) {
                        *m += yy;
                    }
                    count += p.count;
                });
            }
        } else {
            let my_center = my_tree
                .parameters()
                .point_cloud
                .get_point(*my_node.center_index())
                .unwrap();
            for (m, yy) in mom1.iter_mut().zip(my_center) {
                *m += yy;
            }
            for (m, yy) in mom2.iter_mut().zip(my_center) {
                *m += yy * yy;
            }
            count += 1;
        }

        DiagGaussianNode { mom1, mom2, count }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::tree::tests::build_basic_tree;

    #[test]
    fn gaussian_sanity_correct() {
        let basic_tree_data = vec![0.499, 0.49, 0.48, -0.49, 0.0];
        let mom1 = basic_tree_data.iter().fold(0.0, |a, x| a + x);
        let mom2 = basic_tree_data.iter().fold(0.0, |a, x| a + x * x);
        let count = basic_tree_data.len();
        let d = DiagGaussianTree {};
        let mut tree = build_basic_tree();
        tree.add_plugin::<GrandmaDiagGaussian>(d);
        println!("{:?}", tree.reader().len());
        let reader = tree.reader();

        for (si, layer) in tree.reader().layers() {
            println!("Scale Index: {:?}", si);
            layer.for_each_node(|_pi, n| {
                if n.is_leaf() {
                    n.get_plugin_and::<DiagGaussianNode, _, _>(|p| {
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
                        let mut s_mom1 = singles
                            .iter()
                            .map(|i| basic_tree_data[*i as usize])
                            .fold(0.0, |a, x| a + x);
                        let mut s_mom2 = singles
                            .iter()
                            .map(|i| basic_tree_data[*i as usize])
                            .fold(0.0, |a, x| a + x * x);
                        s_mom1 += basic_tree_data[*n.center_index() as usize];
                        s_mom2 += basic_tree_data[*n.center_index() as usize]
                            * basic_tree_data[*n.center_index() as usize];
                        let s_count = singles.len() + 1;
                        println!(
                            "First moment, expected: {:?}, calculated: {:?}",
                            s_mom1, p.mom1[0]
                        );
                        assert_approx_eq!(s_mom1, p.mom1[0]);
                        println!(
                            "Second moment, expected: {:?}, calculated: {:?}",
                            s_mom2, p.mom2[0]
                        );
                        assert_approx_eq!(s_mom2, p.mom2[0]);
                        assert_eq!(s_count, p.count);
                        println!(
                            "=====</Leaf ({},{})>=====",
                            n.scale_index(),
                            n.center_index()
                        );
                    });
                } else {
                    n.get_plugin_and::<DiagGaussianNode, _, _>(|dp| {
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

        reader.get_node_plugin_and::<DiagGaussianNode, _, _>(reader.root_address(), |p| {
            println!(
                "First moment, expected: {:?}, calculated: {:?}",
                mom1, p.mom1[0]
            );
            assert_approx_eq!(mom1, p.mom1[0]);
            println!(
                "Second moment, expected: {:?}, calculated: {:?}",
                mom2, p.mom2[0]
            );
            assert_approx_eq!(mom2, p.mom2[0]);
            assert_eq!(count, p.count);
        });
    }
}
