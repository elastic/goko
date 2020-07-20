//! Interfacees that simplify bulk queries

//use crossbeam_channel::unbounded;
use crate::*;
use rayon::iter::repeatn;


/// Inteface for bulk queries. Handles cloning the readers for you
pub struct BulkInterface<D: PointCloud> {
    reader: CoverTreeReader<D>,
}

impl<D: PointCloud> BulkInterface<D> {
    /// Creates a new one. 
    pub fn new(reader: CoverTreeReader<D>) -> Self {
        BulkInterface {
            reader,
        }
    }

    /// Bulk known path
    pub fn known_path(&self, point_indexes: &[PointIndex]) -> Vec<GokoResult<Vec<(f32,NodeAddress)>>> {
        let indexes_iter = point_indexes.par_chunks(100);
        let reader_copies = indexes_iter.len();
        let mut chunked_results: Vec<Vec<GokoResult<Vec<(f32,NodeAddress)>>>> = indexes_iter
            .zip(repeatn(self.reader.clone(),reader_copies))
            .map(|(chunk_indexes,reader)| {
                chunk_indexes.iter().map(|i| {
                    reader.known_path(*i)
                }).collect()
            }).collect();
        chunked_results.drain(..).fold_first(|mut a, mut x| {a.extend(x.drain(..)); a}).unwrap()
    }

    /// Bulk path
    pub fn path<'a>(&self, points: &[PointRef<'a>]) -> Vec<GokoResult<Vec<(f32,NodeAddress)>>> {
        let point_iter = points.par_chunks(100);
        let reader_copies = point_iter.len();
        let mut chunked_results: Vec<Vec<GokoResult<Vec<(f32,NodeAddress)>>>> = point_iter
            .zip(repeatn(self.reader.clone(),reader_copies))
            .map(|(chunk_points,reader)| {
                chunk_points.iter().map(|p| {
                    reader.path(p)
                }).collect()
            }).collect();

        chunked_results.drain(..).fold_first(|mut a, mut x| {a.extend(x.drain(..)); a}).unwrap()
    }

    /// Bulk knn
    pub fn knn<'a>(&self, points: &[PointRef<'a>], k: usize) -> Vec<GokoResult<Vec<(f32, PointIndex)>>> {
        let point_iter = points.par_chunks(100);
        let reader_copies = point_iter.len();
        let mut chunked_results: Vec<Vec<GokoResult<Vec<(f32,PointIndex)>>>> = point_iter
            .zip(repeatn(self.reader.clone(),reader_copies))
            .map(|(chunk_points,reader)| {
                chunk_points.iter().map(|p| {
                    reader.knn(p,k)
                }).collect()
            }).collect();

        chunked_results.drain(..).fold_first(|mut a, mut x| {a.extend(x.drain(..)); a}).unwrap()
    }

    /// Bulk routing knn
    pub fn routing_knn<'a>(&self, points: &[PointRef<'a>], k: usize) -> Vec<GokoResult<Vec<(f32, PointIndex)>>> {
        let point_iter = points.par_chunks(100);
        let reader_copies = point_iter.len();
        let mut chunked_results: Vec<Vec<GokoResult<Vec<(f32,PointIndex)>>>> = point_iter
            .zip(repeatn(self.reader.clone(),reader_copies))
            .map(|(chunk_points,reader)| {
                chunk_points.iter().map(|p| {
                    reader.routing_knn(p,k)
                }).collect()
            }).collect();

        chunked_results.drain(..).fold_first(|mut a, mut x| {a.extend(x.drain(..)); a}).unwrap()
    }
}