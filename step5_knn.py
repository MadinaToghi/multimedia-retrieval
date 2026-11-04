import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree # Python alternative to ANN

project_dir = Path(__file__).resolve().parent

df = pd.read_csv(project_dir / "normalized_combined_descriptors.csv")

features = df.drop(columns = ['File', 'Class']).to_numpy(dtype=float)

# Build KD-tree for efficient nearest neighbor search
kdtree = cKDTree(features)

query_indices = [10, 5, 80, 56, 33]
k = 6

results = []

# K-nearest neighbors search for each query shape
for query_index in query_indices:

    distances, indices = kdtree.query(features[query_index], k=len(features))

    rank = 1

    for distance, index in zip(distances, indices):
        if index == query_index:
            continue  # Skip the query shape itself

    # Store the results
        results.append({
            'Query_File': df.iloc[query_index]['File'],
            'Query_Class': df.iloc[query_index]['Class'],
            "Rank": rank,
            "File": df.iloc[index]['File'],
            "Class": df.iloc[index]['Class'],
            'Neighbor_Indices': indices.tolist(),
            'Distance': distance
        })

        rank += 1

        if rank > k:
            break  # Stop after finding k neighbors

results_df = pd.DataFrame(results)

results_df.to_csv(project_dir / "knn_results.csv", index=False)

print(results_df)