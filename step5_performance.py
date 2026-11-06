import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from pathlib import Path
import timeit

project_dir = Path(__file__).resolve().parent

# Load normalized feature vectors
df = pd.read_csv(project_dir / "normalized_combined_descriptors.csv")
features = df.drop(columns=["File", "Class"]).to_numpy(dtype=float)

# Build KD-tree
kdtree = cKDTree(features)

# Number of nearest neighbors to test
k_values = range(1, 31)

knn_times = []
brute_times = []

# Brute-force Euclidean search
def brute_force(query_index, k):
    query = features[query_index]

    # Calculate Euclidean distance to every shape
    distances = np.linalg.norm(features - query, axis=1)

    # Exclude the query itself
    distances[query_index] = np.inf

    # Return k nearest neighbor indices
    return np.argpartition(distances, k - 1)[:k]


# KD-tree KNN search
def kdtree_search(query_index, k):
    # Include one extra neighbor because query itself is included
    distances, indices = kdtree.query(features[query_index], k=k+1)

    # Exclude the query itself
    indices = np.atleast_1d(indices)
    indices = [index for index in indices if index != query_index]

    return indices[:k]


# Test both methods for all shapes
for k in k_values:

    # Measure brute-force query time
    brute_time = timeit.timeit(
        lambda: [
            brute_force(i, k) for i in range(len(features))
        ],
        number=3
    )

    # Measure KD-tree query time
    knn_time = timeit.timeit(
        lambda: [
            kdtree_search(i, k) for i in range(len(features))
        ],
        number=3
    )

    # Average time per query in microseconds
    brute_times.append(brute_time / (3 * len(features)) * 1_000_000)
    knn_times.append(knn_time / (3 * len(features)) * 1_000_000)

    print(f"k={k}: brute-force={brute_times[-1]:.2f} µs, "
          f"KD-tree={knn_times[-1]:.2f} µs")


# Save performance measurements
performance_df = pd.DataFrame({
    "k": list(k_values),
    "Brute_force_us": brute_times,
    "KD_tree_us": knn_times
})

performance_df.to_csv(
    project_dir / "step5_performance_results.csv",
    index=False
)

# Create performance plot
plt.figure(figsize=(9, 6))

plt.plot(
    k_values,
    brute_times,
    label="Brute-force Euclidean"
)

plt.plot(
    k_values,
    knn_times,
    label="SciPy cKDTree (exact KNN)"
)

plt.xlabel("Number of nearest neighbors (k)")
plt.ylabel("Average query time (µs)")
plt.title("Average KNN Query Time Over All Shapes")

# Logarithmic scale, like the lecture example
plt.yscale("log")

plt.grid(True)
plt.legend()
plt.tight_layout()

# Save plot
plt.savefig(
    project_dir / "step5_performance_plot.png",
    dpi=200
)

plt.show()

