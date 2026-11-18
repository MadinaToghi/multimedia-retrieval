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
    max_iter=1000,
    n_iter_without_progress=300,
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

fig, ax = plt.subplots(figsize=(16, 10))

scatter = ax.scatter(
    X_2d[:, 0],
    X_2d[:, 1],
    c=class_ids,
    cmap="nipy_spectral",
    s=25,
)

ax.set_title("Interactive t-SNE Visualization of 3D Shapes")
ax.set_xlabel("t-SNE Dimension 1")
ax.set_ylabel("t-SNE Dimension 2")

# Create a scatterplot for highlighting the selected class
highlight = ax.scatter([], [], s=50, zorder=3)

# Create an annotation for the hovered shape
annotation = ax.annotate(
    "",
    xy=(0, 0),
    xytext=(15, 15),
    textcoords="offset points",
    bbox=dict(boxstyle="round", fc="yellow", alpha=0.9)
)
annotation.set_visible(False)

selected_class = None

def on_hover(event):
    global selected_class

    # Check whether the mouse is over a point
    if event.inaxes == ax:
        contains, info = scatter.contains(event)
    else:
        contains = False

    if contains:
        index = info["ind"][0]
        current_class = classes[index]

        # Show filename and class of the specific point
        annotation.xy = X_2d[index]
        annotation.set_text(
            f"Class: {current_class}\n"
            f"File: {files[index]}"
        )
        annotation.set_visible(True)

        # Only update the highlight when the class changes
        if selected_class != current_class:
            selected_class = current_class

            mask = classes == current_class

            # Fade the other classes
            scatter.set_alpha(0.1)

            # Highlight all points from the same class
            highlight.set_offsets(X_2d[mask])
            highlight.set_facecolors(
                scatter.cmap(scatter.norm(class_ids[mask]))
            )

        fig.canvas.draw_idle()

    else:
        # Restore all classes when leaving a point
        if selected_class is not None:
            selected_class = None
            scatter.set_alpha(0.8)
            highlight.set_offsets(np.empty((0, 2)))
            annotation.set_visible(False)
            fig.canvas.draw_idle()


# Detect mouse movement
fig.canvas.mpl_connect("motion_notify_event", on_hover)

handles, _ = scatter.legend_elements(num=None)

ax.legend(
    handles,
    unique_classes,
    title="Shape Classes",
    loc="upper center",
    bbox_to_anchor=(0.5, -0.16),
    fontsize = 8,
    ncol = 8
)

plt.subplots_adjust(bottom=0.42)

tsne_df = combined_df[["Class", "File"]].copy()

tsne_df["TSNE_1"] = X_2d[:, 0]
tsne_df["TSNE_2"] = X_2d[:, 1]

# tsne_embedding.csv = perplexity 30
# tsne_embedding_p50.csv = perplexity 50
# tsne_embedding_p100.csv = perplexity 100
# tsne_embedding_p30_n_iter300.csv = perplexity = 30, n_iter_without_progress = 300
tsne_df.to_csv("tsne_embedding_p30_n_iter300.csv", index=False)

print("Saved t-SNE coordinates.")

plt.savefig(
    "step5_tsne_classes.png",
    dpi=200,
    bbox_inches = "tight"
)

plt.show()