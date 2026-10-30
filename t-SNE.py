import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplcursors

from sklearn.manifold import TSNE

global_df = pd.read_csv("standardized_global_descriptors.csv")
hist_df = pd.read_csv("shape_property_descriptors.csv")

# Merge using the Mesh identity
combined_df = pd.merge (
    global_df,
    hist_df,
    on=["File", "Class"],
    how = "inner",
    validate="one_to_one"
)

# Validate

print("Combined dataset:", combined_df.shape)
print("Number of meshes:", len(combined_df))
print("Duplicates:", combined_df.duplicated(["File", "Class"]).sum())
print("Missing values:", combined_df.isna().sum().sum())


# Verify that all descriptors are finite
feature_columns = combined_df.columns.drop(["Class", "File"])

X = combined_df[feature_columns].astype(float)

print("Feature matrix:", X.shape)
print("All values finite:", np.isfinite(X.to_numpy()).all())

constant_features = combined_df[feature_columns].columns[
    combined_df[feature_columns].nunique() <= 1
].to_list()

print("Constant features:", constant_features)

# Save the normalized feature database
combined_df.to_csv(
    "normalized_combined_descriptors.csv",
    index=False
)

# Use your existing normalized feature matrix
X_values = X.to_numpy()

tsne = TSNE(
    n_components=2,
    perplexity=30,
    random_state=42,
    init="pca",
    learning_rate="auto"
)

X_2d = tsne.fit_transform(X_values)

classes = combined_df["Class"].to_numpy()
files = combined_df["File"].to_numpy()

# Assign each class a color
unique_classes, class_ids = np.unique(
    classes, return_inverse=True
)

fig, ax = plt.subplots(figsize=(12, 9))

scatter = ax.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=class_ids,
    cmap="nipy_spectral",
    s=25,
    alpha=0.8
)

ax.set_title("Interactive t-SNE Visualization of 3D Shapes")
ax.set_xlabel("t-SNE Dimension 1")
ax.set_ylabel("t-SNE Dimension 2")

# Add hover interaction
cursor = mplcursors.cursor(scatter, hover=True)

@cursor.connect("add")
def on_hover(sel):
    index = sel.index

    sel.annotation.set_text(
        f"Class: {classes[index]}\n"
        f"File: {files[index]}"
    )


tsne_df = combined_df[["Class", "File"]].copy()

tsne_df["TSNE_1"] = X_2d[:, 0]
tsne_df["TSNE_2"] = X_2d[:, 1]

tsne_df.to_csv("tsne_embedding.csv", index=False)

print("Saved t-SNE coordinates.")

plt.tight_layout()
plt.show()
