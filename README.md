# Flat KNN

Simple and efficient library for k-nearest neighbors search in Rust.

Library takes advantages of the SIMD capabilities of modern CPUs to perform fast k-NN search. It features a generic API that supports multiple data layouts, numeric types (`f32`, `f16`), and distance metrics without any runtime overhead.

## Usage

```rust
use flat_knn::{knn, Dot, L2};
use half::f16;

fn main() {
    let data = vec![
        1.0, 2.0, 3.0, 4.0, // 0
        5.0, 6.0, 7.0, 8.0, // 1
        9.0, 1.0, 2.0, 3.0, // 2
    ];
    let dim = 4;

    let query = [1.0, 2.0, 3.0, 5.0];
    let k = 2;
    
    // Compute L2
    let neighbors = knn::<_, L2>((&data, dim), &query, k);

    println!("Query: {query:?}");
    println!("{k} nearest neighbors (L2):");
    for (dist, index) in neighbors {
        println!("- Index: {index}, Distance: {dist}");
    }

    // Compute Inner Product
    let neighbors2 = knn::<_, Dot>((&data, dim), &query, k);

    println!("\n{k} nearest neighbors (Inner Product):");
    for (dist, index) in neighbors2 {
        println!("- Index: {index}, Distance: {dist}");
    }
    
    // Work with half-precision (f16) natively using the exact same API
    let data_f16: Vec<f16> = data.iter().map(|&x| f16::from_f32(x)).collect();
    let query_f16: Vec<f16> = query.iter().map(|&x| f16::from_f32(x)).collect();
    let neighbors_f16 = knn::<_, L2>((&data_f16, dim), &query_f16, k);

    println!("\n{k} nearest neighbors (L2 with f16):");
    for (dist, index) in neighbors_f16 {
        println!("- Index: {index}, Distance: {dist}");
    }
}
```

## Extensibility

The `flat-knn` crate relies strictly on static trait bounds, avoiding slow dynamic dispatch or abstraction overhead.

### Extending Distance Metrics

You can swap `L2` or `Dot` with your own distance calculations by implementing the `DistanceMetric<T>` trait on a custom struct.
This trait requires you to define how a calculation builds standard `(distance, index)` metadata into a sorting item for the generic max-heap/min-heap structure:

```rust
use flat_knn::DistanceMetric;
use ordered_float::OrderedFloat;

pub struct CustomMetric;

impl<T: VectorType> DistanceMetric<T> for CustomMetric {
    // Determine heap ordering. For example, smaller distances are better for L2:
    type HeapItem = (OrderedFloat<f32>, usize);

    #[inline(always)]
    fn build_item(query: &[T], chunk: &[T], i: usize) -> Self::HeapItem {
        let dist = /* do math here */;
        (OrderedFloat(dist), i)
    }

    #[inline(always)]
    fn extract_item(item: Self::HeapItem) -> (f32, usize) {
        (item.0.0, item.1)
    }
}
```

### Extending Data Types

By default, the crate implements the `Indexable<T>` trait for 1D vectors partitioned by stride `(Vec<T>, usize)`, nested `Vec<Vec<T>>`, and flat slices `&[T]`.
If you have a custom continuous memory structure like `ndarray` or matrices, you can easily provide a custom target struct implementing the `Indexable<T>` trait, defining `get`, `len`, and `is_empty`.
