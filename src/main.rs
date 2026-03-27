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
    let neighbors = knn::<_, L2>((&data, dim), &query, k);

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
    let neighbors2 = knn::<_, Dot>((&data2, dim2), &query, k);

    println!("\n{k} nearest neighbors (Inner Product):");
    for (dist, index) in neighbors2 {
        println!("- Index: {index}, Distance: {dist}");
    }

    let data_f16: Vec<f16> = data.iter().map(|&x| f16::from_f32(x)).collect();
    let query_f16: Vec<f16> = query.iter().map(|&x| f16::from_f32(x)).collect();
    let neighbors_f16 = knn::<_, L2>((&data_f16, dim), &query_f16, k);

    println!("\n{k} nearest neighbors (L2 with f16):");
    for (dist, index) in neighbors_f16 {
        println!("- Index: {index}, Distance: {dist}");
    }
}
