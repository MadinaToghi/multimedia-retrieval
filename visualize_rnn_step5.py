# Adjust results_path to results from best chosen: KNN or RNN, or distance-based retrieval.
# Each row contains one query and its six nearest neighbors.

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import trimesh

project_dir = Path(__file__).resolve().parent
df = pd.read_csv(project_dir / "rnn_results.csv")

# Folder with all OBJ files
mesh_dir = project_dir / "ShapeDatabase_INFOMR_normalized_cleaned"

# Find all meshes
mesh_files = {p.name: p for p in mesh_dir.rglob("*.obj")}

# Group results by query shape
groups = list(df.groupby(["Query_File", "Query_Class"], sort=False))

# Show only the 6 closest retrieved shapes per query
groups = [
    (key, group.sort_values("Distance").head(6))
    for key, group in groups
]

# Number of rows = number of query shapes
n_rows = len(groups)

max_neighbors = max(len(group) for _, group in groups)
n_cols = 1 + max_neighbors

# Create a 3d plot
fig, axes = plt.subplots(
    n_rows,
    n_cols,
    figsize=(16, 3 * n_rows),
    subplot_kw={"projection": "3d"},
    squeeze=False
)

def draw_mesh(ax, file_name):
    mesh_path = mesh_files.get(file_name)
    
    if not mesh_path:
        ax.set_axis_off()
        return

    # Load the mesh
    mesh = trimesh.load(mesh_path)

    # Get vertices and faces
    vertices = mesh.vertices
    faces = mesh.faces

    ax.plot_trisurf(
        vertices[:, 0],
        vertices[:, 1],
        vertices[:, 2],
        triangles=faces,
        color="red"
    )

    ax.set_xlim(-0.65, 0.65)
    ax.set_ylim(-0.65, 0.65)
    ax.set_zlim(-0.65, 0.65)

     # Keep all shapes at the same visual scale
    ax.set_axis_off()
    ax.view_init(elev=20, azim=45)

    ax.set_anchor("N")


# -------------------------
# 3. Fill the table
# -------------------------
for row, ((query_file, query_class), group) in enumerate(groups):

    group = group.sort_values("Rank")

    # First column = query shape
    ax = axes[row][0]
    draw_mesh(ax, query_file)

    ax.text2D(
        0.5, 0.02,
        f"{query_file}\n{query_class}",
        transform=ax.transAxes,
        ha="center",
        fontsize=8
    )
     # Retrieved neighbors
    for i, (_, neighbor) in enumerate(group.iterrows(), start=1):
        ax = axes[row][i]
        draw_mesh(ax, neighbor["File"])
        ax.text2D(
            0.5, 0.02,
            f'{neighbor["File"]}\n{neighbor["Class"]}\nd={neighbor["Distance"]:.4f}',
            transform=ax.transAxes,
            ha="center",
            fontsize=8
        )
    # Hide unused axes in this row
    for j in range(len(group) + 1, n_cols):
        axes[row][j].set_axis_off()

for col in range(n_cols):
    title = "Query shape" if col == 0 else f"{col:02d}"
    axes[0, col].set_title(title, fontsize=10, pad=12)

plt.subplots_adjust(
    left=0.02,
    right=0.98,
    top=0.92,
    bottom=0.03,
    wspace=0.05,
    hspace=0.15
)

# plt.savefig("ann_rnn_retrieval_table.png")
plt.show()