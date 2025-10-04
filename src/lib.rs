#![feature(binary_heap_into_iter_sorted)]
use ordered_float::OrderedFloat;
use rayon::prelude::*;
use simsimd::SpatialSimilarity;
use std::{cmp::Reverse, collections::BinaryHeap};

pub enum Metric {
    L2,
    Dot,
}

pub trait Indexable: Send + Sync {
    fn get(&self, i: usize) -> &[f32];
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Indexable for (Vec<f32>, usize) {
    fn get(&self, i: usize) -> &[f32] {
        &self.0[i * self.1..(i + 1) * self.1]
    }

    fn len(&self) -> usize {
        self.0.len() / self.1
    }
}

impl Indexable for (&[f32], usize) {
    fn get(&self, i: usize) -> &[f32] {
        &self.0[i * self.1..(i + 1) * self.1]
    }

    fn len(&self) -> usize {
        self.0.len() / self.1
    }
}

impl Indexable for (&Vec<f32>, usize) {
    fn get(&self, i: usize) -> &[f32] {
        &self.0[i * self.1..(i + 1) * self.1]
    }

    fn len(&self) -> usize {
        self.0.len() / self.1
    }
}

impl Indexable for Vec<Vec<f32>> {
    fn get(&self, i: usize) -> &[f32] {
        &self[i]
    }

    fn len(&self) -> usize {
        self.len()
    }
}

impl Indexable for &[Vec<f32>] {
    fn get(&self, i: usize) -> &[f32] {
        &self[i]
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        <[Vec<f32>]>::len(self)
    }
}

impl Indexable for Vec<&[f32]> {
    fn get(&self, i: usize) -> &[f32] {
        self[i]
    }

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    fn len(&self) -> usize {
        Vec::len(self)
    }
}

pub fn knn(data: impl Indexable, query: &[f32], k: usize, metric: Metric) -> Vec<(f32, usize)> {
    match metric {
        Metric::L2 => knn_l2(data, query, k),
        Metric::Dot => knn_dot(data, query, k),
    }
}

fn knn_l2(data: impl Indexable, query: &[f32], k: usize) -> Vec<(f32, usize)> {
    let heap = (0..data.len())
        .into_par_iter()
        .map(|i| {
            let chunk = data.get(i);
            let dist = f32::l2sq(query, chunk).unwrap() as f32;
            (OrderedFloat(dist), i)
        })
        .fold(
            || BinaryHeap::with_capacity(k + 1),
            |mut local_heap, (dist, i)| {
                if local_heap.len() < k {
                    local_heap.push((dist, i));
                } else if let Some(mut top) = local_heap.peek_mut() {
                    if dist < top.0 {
                        *top = (dist, i);
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
                        if item.0 < top.0 {
                            *top = item;
                        }
                    }
                }
                heap1
            },
        );
    let res: Vec<_> = heap.into_iter_sorted().map(|(d, i)| (d.0, i)).collect();
    res.into_iter().rev().collect()
}

fn knn_dot(data: impl Indexable, query: &[f32], k: usize) -> Vec<(f32, usize)> {
    let heap = (0..data.len())
        .into_par_iter()
        .map(|i| {
            let chunk = data.get(i);
            let dist = f32::dot(query, chunk).unwrap() as f32;
            Reverse((OrderedFloat(dist), i))
        })
        .fold(
            || BinaryHeap::with_capacity(k + 1),
            |mut local_heap, item| {
                if local_heap.len() < k {
                    local_heap.push(item);
                } else if let Some(mut top) = local_heap.peek_mut() {
                    if item.0 > top.0 {
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
                        if item.0 > top.0 {
                            *top = item;
                        }
                    }
                }
                heap1
            },
        );
    let res: Vec<_> = heap
        .into_iter_sorted()
        .map(|Reverse((d, i))| (d.0, i))
        .collect();
    res.into_iter().rev().collect()
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
        let neighbors = knn((&data, dim), &query, k, Metric::L2);

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
        let neighbors = knn((&data, dim), &query, k, Metric::Dot);

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
        let pred = knn((&data, DIM), &query, 30, Metric::L2)
            .iter()
            .map(|(_, i)| *i)
            .collect::<Vec<_>>();
        assert_eq!(gt, pred);
    }
}
