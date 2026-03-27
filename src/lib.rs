#![feature(binary_heap_into_iter_sorted)]
use half::f16;
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use simsimd::{f16 as simd_f16, SpatialSimilarity};
use std::{cmp::Reverse, collections::BinaryHeap};

/// A trait defining a specific distance metric for comparing vectors.
///
/// It encapsulates the internal logic for computing the distance between a query and a data chunk.
/// It defines the `HeapItem` used for heap ordering, ensuring that `knn` correctly maintains
/// the most appropriate neighbors irrespective of whether the metric requires a min-heap or max-heap structure.
pub trait DistanceMetric<T>: Send + Sync {
    type HeapItem: Ord + Send + Copy;
    fn build_item(query: &[T], chunk: &[T], i: usize) -> Self::HeapItem;
    fn extract_item(item: Self::HeapItem) -> (f32, usize);
}

/// Represents the L2 (Euclidean) squared distance metric.
///
/// For L2 squared, smaller distances mean higher similarity.
pub struct L2;
impl<T: VectorType> DistanceMetric<T> for L2 {
    type HeapItem = (OrderedFloat<f32>, usize);

    #[inline(always)]
    fn build_item(query: &[T], chunk: &[T], i: usize) -> Self::HeapItem {
        (OrderedFloat(T::l2_squared(query, chunk)), i)
    }

    #[inline(always)]
    fn extract_item(item: Self::HeapItem) -> (f32, usize) {
        (item.0 .0, item.1)
    }
}

/// Represents the Dot Product similarity metric.
///
/// For Dot product, larger values mean higher similarity. A `Reverse` wrapper is used
/// for the `HeapItem` to naturally maintain the correct maximum items within a minimum-backed `BinaryHeap`.
pub struct Dot;
impl<T: VectorType> DistanceMetric<T> for Dot {
    type HeapItem = Reverse<(OrderedFloat<f32>, usize)>;

    #[inline(always)]
    fn build_item(query: &[T], chunk: &[T], i: usize) -> Self::HeapItem {
        Reverse((OrderedFloat(T::dot_product(query, chunk)), i))
    }

    #[inline(always)]
    fn extract_item(item: Self::HeapItem) -> (f32, usize) {
        (item.0 .0 .0, item.0 .1)
    }
}

/// A trait representing a dataset that can be accessed into vector chunks.
///
/// This abstracts over the actual memory layout of the dataset, allowing the `knn` function
/// to seamlessly map over a sliced array `(&[T], dim)`, vectors of vectors `Vec<Vec<T>>`,
/// or arrays of references `Vec<&[T]>`.
pub trait Indexable<T>: Send + Sync {
    fn get(&self, i: usize) -> &[T];
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl<T, C> Indexable<T> for (C, usize)
where
    T: Send + Sync,
    C: AsRef<[T]> + Send + Sync,
{
    #[inline(always)]
    fn get(&self, i: usize) -> &[T] {
        &self.0.as_ref()[i * self.1..(i + 1) * self.1]
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.0.as_ref().len() / self.1
    }
}

impl<T, U> Indexable<T> for &[U]
where
    T: Send + Sync,
    U: AsRef<[T]> + Send + Sync,
{
    #[inline(always)]
    fn get(&self, i: usize) -> &[T] {
        self[i].as_ref()
    }

    #[inline(always)]
    fn len(&self) -> usize {
        <[U]>::len(self)
    }
}

// Blanket implementation for Vec representations, like Vec<Vec<T>> or Vec<&[T]>
impl<T, U> Indexable<T> for Vec<U>
where
    T: Send + Sync,
    U: AsRef<[T]> + Send + Sync,
{
    #[inline(always)]
    fn get(&self, i: usize) -> &[T] {
        self[i].as_ref()
    }

    #[inline(always)]
    fn len(&self) -> usize {
        self.len()
    }
}

// Often passed as reference to Vec
impl<T, U> Indexable<T> for &Vec<U>
where
    T: Send + Sync,
    U: AsRef<[T]> + Send + Sync,
{
    #[inline(always)]
    fn get(&self, i: usize) -> &[T] {
        self[i].as_ref()
    }

    #[inline(always)]
    fn len(&self) -> usize {
        Vec::len(self)
    }
}

/// Performs a K-Nearest Neighbors (KNN) search in parallel.
///
/// This function finds the `k` closest vectors in the `data` to the given `query` vector,
/// using the specified distance metric `M`. It utilizes data parallelization to efficiently
/// compare entries and keeps track of the nearest neighbors.
///
/// # Arguments
///
/// * `data` - The dataset to search against. Must implement `Indexable<T>`. Common usage
///   includes passing a tuple of `(&[T], dim)` for flattened arrays, or passing nested vectors.
/// * `query` - The target query vector.
/// * `k` - The maximum number of nearest neighbors to retrieve.
///
/// # Returns
///
/// A `Vec<(f32, usize)>` sorted from the closest to the furthest distance.
/// Each element contains the calculated distance and the index of the neighbor.
///
/// # Example
///
/// ```
/// use flat_knn::{knn, L2};
///
/// let data = vec![
///     1.0, 2.0, 3.0, 4.0, // index 0 (dist = 1)
///     8.0, 7.0, 6.0, 5.0, // index 1 (dist = 84)
///     1.0, 2.0, 3.0, 9.0, // index 2 (dist = 16)
/// ];
/// let dim = 4;
/// let query = [1.0, 2.0, 3.0, 5.0];
///
/// // Find 2 nearest neighbors using L2 distance
/// let neighbors = knn::<_, L2>((&data, dim), &query, 2);
///
/// assert_eq!(neighbors.len(), 2);
/// assert_eq!(neighbors[0], (1.0, 0));
/// assert_eq!(neighbors[1], (16.0, 2));
/// ```
pub fn knn<T: VectorType, M: DistanceMetric<T>>(
    data: impl Indexable<T>,
    query: &[T],
    k: usize,
) -> Vec<(f32, usize)> {
    let heap = (0..data.len())
        .into_par_iter()
        .map(|i| M::build_item(query, data.get(i), i))
        .fold(
            || BinaryHeap::with_capacity(k + 1),
            |mut local_heap, item| {
                if local_heap.len() < k {
                    local_heap.push(item);
                } else if let Some(mut top) = local_heap.peek_mut() {
                    if item < *top {
                        *top = item;
                    }
                }
                local_heap
            },
        )
        .reduce(
            || BinaryHeap::with_capacity(k + 1),
            |mut heap1, heap2| {
                for item in heap2.into_iter() {
                    if heap1.len() < k {
                        heap1.push(item);
                    } else if let Some(mut top) = heap1.peek_mut() {
                        if item < *top {
                            *top = item;
                        }
                    }
                }
                heap1
            },
        );

    let res: Vec<_> = heap.into_iter_sorted().map(M::extract_item).collect();
    res.into_iter().rev().collect()
}

pub trait VectorType: Send + Sync + Sized {
    fn l2_squared(query: &[Self], chunk: &[Self]) -> f32;
    fn dot_product(query: &[Self], chunk: &[Self]) -> f32;
}

impl VectorType for f32 {
    #[inline(always)]
    fn l2_squared(query: &[f32], chunk: &[f32]) -> f32 {
        f32::l2sq(query, chunk).unwrap() as f32
    }

    #[inline(always)]
    fn dot_product(query: &[f32], chunk: &[f32]) -> f32 {
        f32::dot(query, chunk).unwrap() as f32
    }
}

#[inline(always)]
fn as_simsimd_slice(slice: &[f16]) -> &[simd_f16] {
    // SAFETY: Both types are exactly 16 bits (u16) under the hood
    // and represent the identical IEEE 754 half-precision format.
    unsafe { std::slice::from_raw_parts(slice.as_ptr() as *const simd_f16, slice.len()) }
}

impl VectorType for f16 {
    #[inline(always)]
    fn l2_squared(query: &[f16], chunk: &[f16]) -> f32 {
        simd_f16::l2sq(as_simsimd_slice(query), as_simsimd_slice(chunk)).unwrap() as f32
    }

    #[inline(always)]
    fn dot_product(query: &[f16], chunk: &[f16]) -> f32 {
        simd_f16::dot(as_simsimd_slice(query), as_simsimd_slice(chunk)).unwrap() as f32
    }
}

#[cfg(test)]
mod tests {
    use linfa_nn::{distance::L2Dist, LinearSearch, NearestNeighbour};
    use ndarray::Array2;
    use rand::{distr::Uniform, Rng};

    use super::*;

    const DIM: usize = 768;

    fn generate_random_data(num_vectors: usize) -> Vec<f32> {
        let mut rng = rand::rng();
        let uniform = Uniform::new(0.0, 10.0).unwrap();
        let mut data = Vec::with_capacity(num_vectors * DIM);

        for _ in 0..(num_vectors * DIM) {
            data.push(rng.sample(uniform));
        }
        data
    }

    #[test]
    fn test_knn_search_l2() {
        let data = vec![
            1.0, 2.0, 3.0, 4.0, // 0: dist = 1
            8.0, 7.0, 6.0, 5.0, // 1: dist = 84
            1.0, 2.0, 3.0, 9.0, // 2: dist = 16
            4.0, 3.0, 2.0, 1.0, // 3: dist = 27
        ];
        let dim: usize = 4;
        let query = [1.0, 2.0, 3.0, 5.0];
        let k = 4;
        let neighbors = knn::<_, L2>((&data, dim), &query, k);

        assert_eq!(neighbors.len(), 4);
        assert_eq!(neighbors[0], (1.0, 0));
        assert_eq!(neighbors[1], (16.0, 2));
    }

    #[test]
    fn test_knn_search_dot() {
        let data = vec![1.0, -0.1, 0.3, 1.0, -1.0, 0.0, 0.0, -1.0];
        let dim = 2;
        let query = [0.9, 0.1];
        let k = 4;
        let neighbors = knn::<_, Dot>((&data, dim), &query, k);

        assert_eq!(neighbors.len(), 4);
        assert_eq!(neighbors[0], (0.89, 0));
        assert_eq!(neighbors[1], (0.37, 1));
        assert_eq!(neighbors[2], (-0.1, 3));
        assert_eq!(neighbors[3], (-0.9, 2));
    }

    #[test]
    fn test_compare_with_linfa_l2() {
        let num_vectors = 10_000;
        let data = generate_random_data(num_vectors);
        let dataset = Array2::from_shape_vec((num_vectors, DIM), data.clone())
            .expect("Failed to reshape data.");
        let query = generate_random_data(1);

        let index = LinearSearch::new()
            .from_batch(&dataset, L2Dist {})
            .expect("Failed to build LinearSearch index");

        let gt = index
            .k_nearest((&query).into(), 30)
            .unwrap()
            .iter()
            .map(|(_, i)| *i)
            .collect::<Vec<_>>();
        let pred = knn::<_, L2>((&data, DIM), &query, 30)
            .iter()
            .map(|(_, i)| *i)
            .collect::<Vec<_>>();
        assert_eq!(gt, pred);
    }

    #[test]
    fn test_knn_f16_l2() {
        let data = vec![
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(4.0),
            f16::from_f32(8.0),
            f16::from_f32(7.0),
            f16::from_f32(6.0),
            f16::from_f32(5.0),
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(9.0),
        ];
        let dim = 4;
        let query = [
            f16::from_f32(1.0),
            f16::from_f32(2.0),
            f16::from_f32(3.0),
            f16::from_f32(5.0),
        ];
        let k = 2;
        let neighbors = knn::<_, L2>((&data, dim), &query, k);

        assert_eq!(neighbors.len(), 2);
        assert_eq!(neighbors[0].1, 0); // closest is the first one
        assert_eq!(neighbors[1].1, 2); // second is the third one
    }

    #[test]
    fn test_knn_f16_dot() {
        let data = vec![
            f16::from_f32(1.0),
            f16::from_f32(-0.1),
            f16::from_f32(0.3),
            f16::from_f32(1.0),
            f16::from_f32(-1.0),
            f16::from_f32(0.0),
            f16::from_f32(0.0),
            f16::from_f32(-1.0),
        ];
        let dim = 2;
        let query = [f16::from_f32(0.9), f16::from_f32(0.1)];
        let k = 4;
        let neighbors = knn::<_, Dot>((&data, dim), &query, k);

        assert_eq!(neighbors.len(), 4);
        assert_eq!(neighbors[0].1, 0); // 0.89
        assert_eq!(neighbors[1].1, 1); // 0.37
        assert_eq!(neighbors[2].1, 3); // -0.1
        assert_eq!(neighbors[3].1, 2); // -0.9
    }
}
