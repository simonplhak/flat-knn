# Flat KNN

Simple and efficient library for k-nearest neighbors search in Rust.

Library takes advantages of the SIMD capabilities of modern CPUs to perform fast k-NN search. 

## Usage
```rust
use flat_knn::{knn, Metric};

fn main() {
    let data = vec![
        1.0, 2.0, 3.0, 4.0, // 0
        5.0, 6.0, 7.0, 8.0, // 1
        9.0, 1.0, 2.0, 3.0, // 2
    ];
    let dim = 4;

    let query = [1.0, 2.0, 3.0, 5.0];
    let k = 2;
    let neighbors = knn((&data, dim), &query, k, Metric::L2);

    println!("Query: {query:?}");
    println!("{k} nearest neighbors (L2):");
    for (dist, index) in neighbors {
        println!("- Index: {index}, Distance: {dist}");
    }

    let data2 = vec![
        1.0, 2.0, 3.0, 4.0, // 0
        5.0, 6.0, 7.0, 8.0, // 1
        9.0, 1.0, 2.0, 3.0, // 2
    ];
    let dim2 = 4;
    let neighbors2 = knn((&data2, dim2), &query, k, Metric::Dot);

    println!("\n{k} nearest neighbors (Inner Product):");
    for (dist, index) in neighbors2 {
        println!("- Index: {index}, Distance: {dist}");
    }
}
```