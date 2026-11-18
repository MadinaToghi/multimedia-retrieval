import pandas as pd
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree # Python alternative to ANN

project_dir = Path(__file__).resolve().parent
roc_output_dir = project_dir / "roc"
roc_output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(project_dir / "normalized_combined_descriptors.csv")

features = df.drop(columns = ['File', 'Class']).to_numpy(dtype=float)

# Build KD-tree for efficient nearest neighbor search
kdtree = cKDTree(features)

r = 1.5 # Search radius

results = []
all_results = []

# K-nearest neighbors search for each query shape
for query_index in df.index:

    query = features[query_index]

    all_distances, all_indices = kdtree.query(query, k=len(features))
    rank = 1
    for distance, index in zip(all_distances, all_indices):
        if index == query_index:
            continue

        all_results.append({
            'Query_File': df.iloc[query_index]['File'],
            'Query_Class': df.iloc[query_index]['Class'],
            "Rank": rank,
            "File": df.iloc[index]['File'],
            "Class": df.iloc[index]['Class'],
            'Distance': distance
        })
        rank += 1

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
all_results_df = pd.DataFrame(all_results)

results_df.to_csv(project_dir / "rnn_results.csv", index=False)
all_results_df.to_csv(roc_output_dir / "rnn_all_distances.csv", index=False)

print(f"Wrote {len(results_df)} radius-{r} neighbor rows to rnn_results.csv")
print(
    f"Wrote {len(all_results_df)} full-gallery distance rows to "
    f"{roc_output_dir.name}/rnn_all_distances.csv"
)