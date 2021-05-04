//! Interfacees that simplify bulk queries

//use crossbeam_channel::unbounded;
use crate::*;
use ndarray::ArrayView2;
use rayon::iter::repeatn;
use std::ops::Deref;

/// Inteface for bulk queries. Handles cloning the readers for you
pub struct BulkInterface<D: PointCloud> {
    reader: CoverTreeReader<D>,
}

impl<D: PointCloud> BulkInterface<D> {
    /// Creates a new one.
    pub fn new(reader: CoverTreeReader<D>) -> Self {
        BulkInterface { reader }
    }

    /// Applies the passed in fn to the passed in indexes and collects the result in a vector. Core function for this struct.
    pub fn index_map_with_reader<F, T>(&self, point_indexes: &[usize], f: F) -> Vec<T>
    where
        F: Fn(&CoverTreeReader<D>, usize) -> T + Send + Sync,
        T: Send + Sync,
    {
        let indexes_iter = point_indexes.par_chunks(100);
        let reader_copies = indexes_iter.len();
        let mut chunked_results: Vec<Vec<T>> = indexes_iter
            .zip(repeatn(self.reader.clone(), reader_copies))
            .map(|(chunk_indexes, reader)| chunk_indexes.iter().map(|p| f(&reader, *p)).collect())
            .collect();
        chunked_results
            .drain(..)
            .reduce(|mut a, mut x| {
                a.extend(x.drain(..));
                a
            })
            .unwrap()
    }

    /// Applies the passed in fn to the passed in indexes and collects the result in a vector. Core function for this struct.
    pub fn point_map_with_reader<P: Deref<Target = D::Point> + Send + Sync, F, T>(
        &self,
        points: &[P],
        f: F,
    ) -> Vec<T>
    where
        F: Fn(&CoverTreeReader<D>, &P) -> T + Send + Sync,
        T: Send + Sync,
    {
        let point_iter = points.par_chunks(100);
        let reader_copies = point_iter.len();
        let mut chunked_results: Vec<Vec<T>> = point_iter
            .zip(repeatn(self.reader.clone(), reader_copies))
            .map(|(chunk_points, reader)| chunk_points.iter().map(|p| f(&reader, p)).collect())
            .collect();
        chunked_results
            .drain(..)
            .reduce(|mut a, mut x| {
                a.extend(x.drain(..));
                a
            })
            .unwrap()
    }

    /// Bulk known path
    pub fn known_path(&self, point_indexes: &[usize]) -> Vec<GokoResult<Vec<(f32, NodeAddress)>>> {
        self.index_map_with_reader(point_indexes, |reader, i| reader.known_path(i))
    }

    /// Bulk known path
    pub fn known_path_and<F, T>(&self, point_indexes: &[usize], f: F) -> Vec<T>
    where
        F: Fn(&CoverTreeReader<D>, GokoResult<Vec<(f32, NodeAddress)>>) -> T + Send + Sync,
        T: Send + Sync,
    {
        self.index_map_with_reader(point_indexes, |reader, i| f(&reader, reader.known_path(i)))
    }

    /// Bulk known path
    pub fn path<P: Deref<Target = D::Point> + Send + Sync>(
        &self,
        points: &[P],
    ) -> Vec<GokoResult<Vec<(f32, NodeAddress)>>> {
        self.point_map_with_reader(points, |reader, p| reader.path(p))
    }

    /// Bulk knn
    pub fn knn<P: Deref<Target = D::Point> + Send + Sync>(
        &self,
        points: &[P],
        k: usize,
    ) -> Vec<GokoResult<Vec<(f32, usize)>>> {
        self.point_map_with_reader(points, |reader, p| reader.knn(p, k))
    }

    /// Bulk routing knn
    pub fn routing_knn<P: Deref<Target = D::Point> + Send + Sync>(
        &self,
        points: &[P],
        k: usize,
    ) -> Vec<GokoResult<Vec<(f32, usize)>>> {
        self.point_map_with_reader(points, |reader, p| reader.routing_knn(p, k))
    }
}

impl<D: PointCloud<Point = [f32]>> BulkInterface<D> {
    /// Applies the passed in fn to the passed in indexes and collects the result in a vector. Core function for this struct.
    pub fn array_map_with_reader<'a, F, T>(&self, points: ArrayView2<'a, f32>, f: F) -> Vec<T>
    where
        F: Fn(&CoverTreeReader<D>, &&[f32]) -> T + Send + Sync,
        T: Send + Sync,
    {
        let indexes: Vec<usize> = (0..points.nrows()).collect();
        let point_iter = indexes.par_chunks(100);
        let reader_copies = point_iter.len();

        let mut chunked_results: Vec<Vec<T>> = point_iter
            .zip(repeatn(self.reader.clone(), reader_copies))
            .map(|(chunk_points, reader)| {
                chunk_points
                    .iter()
                    .map(|i| f(&reader, &points.row(*i).as_slice().unwrap()))
                    .collect()
            })
            .collect();
        chunked_results
            .drain(..)
            .reduce(|mut a, mut x| {
                a.extend(x.drain(..));
                a
            })
            .unwrap()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::env;

    use crate::covertree::tests::build_mnist_tree;

    #[test]
    fn bulk_path() {
        if env::var("TRAVIS_RUST_VERSION").is_err() {
            let tree = build_mnist_tree();
            let reader = tree.reader();
            let interface = BulkInterface::new(tree.reader());
            let cloud = reader.point_cloud();

            let points: Vec<&[f32]> = (0..100).map(|i| cloud.point(i).unwrap()).collect();

            let path_results = interface.path(&points);
            for (i, path) in path_results.iter().enumerate() {
                let old_path = reader.path(&cloud.point(i).unwrap()).unwrap();
                for ((d1, a1), (d2, a2)) in (path.as_ref().unwrap()).iter().zip(old_path) {
                    assert_approx_eq!(*d1, d2);
                    assert_eq!(*a1, a2);
                }
            }
        }
    }

    #[test]
    fn bulk_knn() {
        if env::var("TRAVIS_RUST_VERSION").is_err() {
            let tree = build_mnist_tree();
            let reader = tree.reader();
            let interface = BulkInterface::new(tree.reader());
            let cloud = reader.point_cloud();

            let points: Vec<&[f32]> = (0..10).map(|i| cloud.point(i).unwrap()).collect();

            let knn_results = interface.knn(&points, 5);
            for (i, knn) in knn_results.iter().enumerate() {
                let old_knn = reader.knn(&cloud.point(i).unwrap(), 5).unwrap();
                for ((d1, a1), (d2, a2)) in (knn.as_ref().unwrap()).iter().zip(old_knn) {
                    assert_approx_eq!(*d1, d2);
                    assert_eq!(*a1, a2);
                }
            }
        }
    }
}
