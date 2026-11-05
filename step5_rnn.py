import pandas as pd
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree # Python alternative to ANN

project_dir = Path(__file__).resolve().parent

df = pd.read_csv(project_dir / "normalized_combined_descriptors.csv")

features = df.drop(columns = ['File', 'Class']).to_numpy(dtype=float)

# Build KD-tree for efficient nearest neighbor search
kdtree = cKDTree(features)

r = 1.5 # Search radius

results = []

# K-nearest neighbors search for each query shape
for query_index in df.index:

    query = features[query_index]

    indices = kdtree.query_ball_point(query, r=r)

    indices = [index for index in indices if index != query_index]  # Exclude the query shape itself

    distances = [
        np.linalg.norm(query - features[index]) for index in indices
    ]

    neighbors = sorted(zip(distances, indices))

    for rank, (distance, index) in enumerate(neighbors, start=1):

    # Store the results
        results.append({
            'Query_File': df.iloc[query_index]['File'],
            'Query_Class': df.iloc[query_index]['Class'],
            "Rank": rank,
            "File": df.iloc[index]['File'],
            "Class": df.iloc[index]['Class'],
            'Distance': distance
        })

results_df = pd.DataFrame(results)

results_df.to_csv(project_dir / "rnn_results.csv", index=False)

print(results_df)