import numpy as np
import faiss
import time

NUM_VECTORS = 10_000
DIM = 768
K = 10


def generate_random_data(num_vectors, dim):
    return np.random.uniform(0.0, 10.0, size=(num_vectors, dim)).astype('float32')


def benchmark_faiss_knn(num_vectors=NUM_VECTORS, dim=DIM, k=K, n_runs=10):
    data = generate_random_data(num_vectors, dim)
    query = generate_random_data(1, dim)


    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        D, I = faiss.knn(query, data, k, faiss.METRIC_L2)
        end = time.perf_counter()
        times.append(end - start)

    print(f"faiss.IndexFlatL2.search: avg {np.mean(times)*1e3:.3f} ms over {n_runs} runs (min {np.min(times)*1e3:.3f} ms, max {np.max(times)*1e3:.3f} ms)")


if __name__ == "__main__":
    benchmark_faiss_knn()
