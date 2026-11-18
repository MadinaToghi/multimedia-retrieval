import pandas as pd
from pathlib import Path
from scipy.spatial import cKDTree # Python alternative to ANN

project_dir = Path(__file__).resolve().parent
roc_output_dir = project_dir / "roc"
roc_output_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(project_dir / "normalized_combined_descriptors.csv")

features = df.drop(columns = ['File', 'Class']).to_numpy(dtype=float)

# Build KD-tree for efficient nearest neighbor search
kdtree = cKDTree(features)

k = 6

results = []
all_results = []

# K-nearest neighbors search for each query shape
for query_index in df.index:

    distances, indices = kdtree.query(features[query_index], k=len(features))

    rank = 1

    for distance, index in zip(distances, indices):
        if index == query_index:
            continue  # Skip the query shape itself

        neighbor = {
            'Query_File': df.iloc[query_index]['File'],
            'Query_Class': df.iloc[query_index]['Class'],
            "Rank": rank,
            "File": df.iloc[index]['File'],
            "Class": df.iloc[index]['Class'],
            'Distance': distance
        }
        all_results.append(neighbor)

        if rank <= k:
            results.append({
                **neighbor,
                'Neighbor_Indices': indices.tolist(),
            })

        rank += 1

results_df = pd.DataFrame(results)
all_results_df = pd.DataFrame(all_results)

results_df.to_csv(project_dir / "knn_results.csv", index=False)
all_results_df.to_csv(roc_output_dir / "knn_all_distances.csv", index=False)

print(f"Wrote {len(results_df)} top-{k} neighbor rows to knn_results.csv")
print(
    f"Wrote {len(all_results_df)} full-gallery distance rows to "
    f"{roc_output_dir.name}/knn_all_distances.csv"
)