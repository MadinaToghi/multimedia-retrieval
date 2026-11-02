# python step5_ann.py

# Step 5: Spatial Search using the Approximate Nearest Neighbor (ANN) library
#
# This script:
# 1. Loads the compiled C++ ANN library using ctypes.
# 2. Loads the feature vectors from the normalized feature database.
# 3. Builds a KD-tree to accelerate nearest-neighbor searching.
# 4. Retrieves the K nearest shapes for a given query shape.
# 5. Validates ANN results against brute-force Euclidean distance.
# 6. Saves retrieval results to a CSV file for later visualization and comparison.

import ctypes
from pathlib import Path

import numpy as np
import pandas as pd

# Load the ANN C++ library 

libname = Path(__file__).resolve().parent / "annlib.dll"
c_lib = ctypes.CDLL(str(libname))

print("ANN library loaded successfully!")

# Load the feature dataset

data_path = Path(__file__).resolve().parent / "normalized_combined_descriptors.csv"
df = pd.read_csv(data_path)

feature_columns = df.columns.drop(['File', 'Class'])

# Convert the feature matrix to a contiguous NumPy array of type float64 for compatibility with the C++ library
X = np.ascontiguousarray(
    df[feature_columns].to_numpy(dtype=np.float64)
)

n_shapes, n_features = X.shape

print("Number of shapes:", n_shapes)
print("Number of features:", n_features)

# Validate the feature matrix
if not np.isfinite(X).all():
    raise ValueError("Feature matrix contains non-finite values.")

if n_shapes < 2:
    raise ValueError("Feature matrix must contain at least two shapes.")

# Configure the C++ interface

# Define pointer types for the C++ functions
double_ptr = ctypes.POINTER(ctypes.c_double)
int_ptr = ctypes.POINTER(ctypes.c_int)

# Define the argument and return types for building the ANN index 
# The C++ function expects:
c_lib.build_ann_index.argtypes = [
    double_ptr, # Pointer to the feature matrix
    ctypes.c_int, # Number of shapes (rows)
    ctypes.c_int  # Number of features (columns)
    ]

c_lib.build_ann_index.restype = ctypes.c_void_p # Returns a pointer to the ANN index

# K-nearest neighbors search function signature
# The C++ function receives the ANN index, query vector,
# number of neighbors, approximation parameter,
# and arrays for storing the retrieved indices and distances.
c_lib.knn_search.argtypes = [
    ctypes.c_void_p, # Pointer to the ANN index
    double_ptr, # Pointer to the query vector
    ctypes.c_int,  # Number of neighbors to search for
    ctypes.c_double, # Approximation parameter (eps)
    int_ptr, # Output neighbor indices
    double_ptr # Output distances
    ]

c_lib.knn_search.restype = None # No return value, results are written to the provided pointers

# Define the function used to release allocated C++ memory.
# The ANN KD-tree is stored in C++ memory and must be freed
# after all searches are completed.
c_lib.free_ann_index.argtypes = [
    ctypes.c_void_p
]

c_lib.free_ann_index.restype = None

# Build ANN KD-tree 

# Construct a KD-tree using all feature vectors.
# The tree organizes the 132-dimensional feature space
# to support more efficient nearest-neighbor searches.
ann_index = c_lib.build_ann_index(
    X.ctypes.data_as(double_ptr),
    n_shapes,
    n_features
)

# Check whether the ANN index was created successfully.
if not ann_index:
    raise RuntimeError("Failed to build ANN index.")

print("ANN index built successfully!")

# Implement K-nearest neighbor search

def knn_search(query_index, k=6, eps=0.0):
    """
    Perform K-nearest neighbor search using the ANN index.

    Parameters:
    - query_index: Index of the query shape in the dataset.
    - k: Number of nearest neighbors to retrieve.
    - eps: Approximation parameter 
           0.0 = exact search.
           >0.0 = approximate search.

    Returns:
        DataFrame containing filenames, classes and distances (Euclidean distance to the query shape).
    """

    # Check that the query index exists in the database.
    if not 0 <= query_index < n_shapes:
        raise ValueError("Invalid query index.")

    # K cannot exceed the number of other shapes in the database.
    if not 1 <= k < n_shapes:
        raise ValueError("K must be between 1 and n_shapes - 1.")
    
     # The approximation parameter must be a valid non-negative value.
    if eps < 0 or not np.isfinite(eps):
        raise ValueError("eps must be finite and non-negative.")

    # Select the query shape's feature vector
    query_vector = np.ascontiguousarray(X[query_index])

    count = k + 1  # Request k+1 because query itself is in the database

    # Allocate arrays for search results
    indices = np.empty(count, dtype=np.int32)
    distances = np.empty(count, dtype=np.float64)

    # Perform KNN search using the ANN C++ library
    # The wrapper converts ANN's squared distances to
    # ordinary Euclidean distances.
    c_lib.knn_search(
        ann_index,
        query_vector.ctypes.data_as(double_ptr),
        count,
        eps,
        indices.ctypes.data_as(int_ptr),
        distances.ctypes.data_as(double_ptr)
    )

    # Store the retrieved shapes and their distances.
    results = []

    # Collect the retrieved shapes
    for idx, distance in zip(indices, distances):

        # Exclude the query shape itself
        if idx == query_index:
            continue

        results.append({
            "File": df.iloc[idx]["File"],
            "Class": df.iloc[idx]["Class"],
            "Distance": float(distance)
        })

        if len(results) == k:
            break

    # Return the results as a DataFrame
    return pd.DataFrame(
        results,
        columns=["File", "Class", "Distance"]
    )


# Test the KNN search 

try:
    query_indices = [10, 5, 80, 56, 33] # Adjust based on step 4 choices
    k = 6

    # Store the results of all queries for CSV export.
    all_results = []

    for query_index in query_indices:

        print("\nQuery shape:")
        print(df.iloc[query_index][["File", "Class"]])

        results = knn_search(query_index, k=k, eps=0.0)

        # Store query information with retrieval results
        results_to_save = results.copy()

        results_to_save.insert(0, "Query_File", df.iloc[query_index]["File"])
        results_to_save.insert(1, "Query_Class", df.iloc[query_index]["Class"])

        # Assign a rank to each neighbor, starting at 1.
        # Rank 1 represents the closest retrieved shape.
        results_to_save.insert(2, "Rank", range(1, len(results_to_save) + 1))

        # Add this query's results to the complete list.
        all_results.append(results_to_save)

        print(f"\n{k} nearest neighbors:")
        print(results)

        # Verify ANN results with brute-force Euclidean distance
        all_distances = np.linalg.norm(
            X - X[query_index],
            axis=1
        )

        # Exclude the query shape itself
        all_distances[query_index] = np.inf

        # Retrieve the K closest shapes
        expected_indices = np.argsort(all_distances)[:k]

        # Extract filenames from both retrieval methods.
        expected_files = df.iloc[expected_indices]["File"].tolist()
        ann_files = results["File"].tolist()

        print("\nBrute-force nearest neighbors:")
        print(expected_files)

        print("\nANN nearest neighbors:")
        print(ann_files)

        # Check whether ANN and brute-force search return
        # the same K shapes in the same order.
        print("\nResults match:", expected_files == ann_files)

    # Combine retrieval results from all queries
    combined_results = pd.concat(all_results, ignore_index=True)

    # Save results to CSV
    output_path = Path(__file__).resolve().parent / "ann_knn_results.csv"

    combined_results.to_csv(output_path, index=False)

    print(f"\nKNN results saved to: {output_path}")

# Release allocated ANN memory

finally: 
    # Free the C++ memory used by the ANN KD-tree.
    # The finally block ensures cleanup even if
    # an error occurs during retrieval or CSV export.
    
    c_lib.free_ann_index(ann_index)
    print("\nANN index memory released!")