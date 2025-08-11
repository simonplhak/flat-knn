#![feature(test)]

extern crate blas_src;
extern crate test;

use faiss::*;
use flat_knn::knn;
use rand::{distr::Uniform, Rng};

use ndarray::{Array1, Array2, Axis};

const NUM_VECTORS: usize = 10_000;
const DIM: usize = 768;
const K: usize = 30;

fn generate_random_ndarray(num_vectors: usize) -> Array2<f32> {
    let mut rng = rand::rng();
    let uniform = Uniform::new(0.0, 10.0).unwrap();
    let data: Vec<f32> = (0..(num_vectors * DIM))
        .map(|_| rng.sample(uniform))
        .collect();
    Array2::from_shape_vec((num_vectors, DIM), data).unwrap()
}

fn generate_random_data(num_vectors: usize) -> Vec<f32> {
    let mut rng = rand::rng();
    let uniform = Uniform::new(0.0, 10.0).unwrap();
    let mut data = Vec::with_capacity(num_vectors * DIM);

    for _ in 0..(num_vectors * DIM) {
        data.push(rng.sample(uniform));
    }
    data
}

fn l2_distances(data: &Array2<f32>, query: &Array1<f32>) -> Array1<f32> {
    let data_norms = data.map_axis(Axis(1), |row| row.dot(&row));
    let query_norm = query.dot(query);
    let dots = data.dot(query);
    data_norms + query_norm - 2.0 * dots
}

fn knn_ndarray(data: &Array2<f32>, query: &Array1<f32>, k: usize) -> Array1<usize> {
    let dists = l2_distances(data, query);
    let mut idx: Vec<usize> = (0..dists.len()).collect();
    idx.select_nth_unstable_by(k, |&a, &b| dists[a].partial_cmp(&dists[b]).unwrap());
    Array1::from(idx[..k].to_vec())
}

#[bench]
fn benchmark_my_search_l2(b: &mut test::Bencher) {
    let data: Vec<f32> = generate_random_data(NUM_VECTORS);
    let query: Vec<f32> = generate_random_data(1);

    b.iter(|| {
        knn(
            test::black_box((&data, DIM)),
            test::black_box(&query),
            K,
            flat_knn::Metric::L2,
        );
    });
}

#[bench]
fn benchmark_ndarray_knn_blas(b: &mut test::Bencher) {
    let data = generate_random_ndarray(NUM_VECTORS);
    let query = generate_random_ndarray(1).row(0).to_owned();
    b.iter(|| {
        let _ = knn_ndarray(&data, test::black_box(&query), test::black_box(K));
    });
}

#[bench]
fn benchmark_faiss_search(b: &mut test::Bencher) {
    use faiss::*;
    let data: Vec<f32> = generate_random_data(NUM_VECTORS);
    let mut index = index_factory(DIM as u32, "Flat", MetricType::L2).unwrap();

    index.add(&data).unwrap();

    let query: Vec<f32> = generate_random_data(1);

    b.iter(|| {
        index
            .search(test::black_box(&query), test::black_box(K))
            .unwrap();
    });
}

#[bench]
fn benchmark_faiss_hnsw32_search(b: &mut test::Bencher) {
    let data: Vec<f32> = generate_random_data(NUM_VECTORS);

    let mut index = index_factory(DIM as u32, "HNSW32", MetricType::L2).unwrap();
    index.add(&data).unwrap();

    let query: Vec<f32> = generate_random_data(1);

    b.iter(|| {
        index
            .search(test::black_box(&query), test::black_box(K))
            .unwrap();
    });
}

#[bench]
fn benchmark_faiss_hnsw32_sq8_search(b: &mut test::Bencher) {
    let data: Vec<f32> = generate_random_data(NUM_VECTORS);

    let mut index = index_factory(DIM as u32, "HNSW32_SQ8", MetricType::L2).unwrap();

    let start = std::time::Instant::now();
    index.train(&data).unwrap();
    let duration = start.elapsed();
    println!("HNSW32_SQ8 training time: {}ms", duration.as_millis());

    let query: Vec<f32> = generate_random_data(1);

    b.iter(|| {
        index
            .search(test::black_box(&query), test::black_box(K))
            .unwrap();
    });
}
