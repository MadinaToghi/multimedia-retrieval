import trimesh as tr
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from vedo import Mesh, show

data_path = Path("../ShapeDatabase_INFOMR")

obj_files = list(data_path.rglob("*.obj"))

random.seed(43)
# Group files by category (parent folder name) and sample 10 files from each category (Disproportionate stratified sampling)
# Parent folder name is taken per file, from each parent folder 10 random files are taken
# If there are only 10 (or less than 10) files in a category, all files are taken
# Sample files are stored in sample_files list without parent folder name
files_by_category = {}

for path in obj_files:
    files_by_category.setdefault(path.parent.name, []).append(path)

sample_files = []

for category, files in files_by_category.items():
    sample_files.extend(random.sample(files, min(10, len(files))))

def show_mesh(mesh, path):
    # Convert to Vedo for visualization
    vedo_mesh = Mesh([mesh.vertices, mesh.faces])
    # Makes sure objects are shaded
    vedo_mesh.c("lightblue")
    vedo_mesh.compute_normals()

    # Cell edges
    edges = vedo_mesh.clone().wireframe().c("black").lw(1)

    # Interactive viewer
    show(
        vedo_mesh,
        # Remove edges for a cleaner look
        edges, 
        title=path.name, 
        axes=1, 
        viewup="z", 
        interactive=True,
        # True = Viewer will open in a new window each time. Set to False to reuse the same window.
        new=False
    )

# Trimesh makes all shapes into triangles, so we need to read the obj files directly to determine if they are triangles, quads, or mixed.
def get_face_types(obj_path):
    face_vertex_counts = set()
    face_count = 0

    with open(obj_path, 'r') as f:
        for line in f:
            if line.startswith('f '):
                face_count += 1
                n = len(line.strip().split()[1:])
                face_vertex_counts.add(n)
                # once we've seen both 3 and 4, no need to keep scanning
                if len(face_vertex_counts) > 1:
                    break
    return face_count, face_vertex_counts

# Check if the mesh is made of triangles, quads, or mixed 
# Technically, face vertex counts can be more than 4, like hexagons etc, however, we only check for triangles quads or mixed
def classify(face_vertex_counts):
    if face_vertex_counts == {3}:
        return "triangles"
    elif face_vertex_counts == {4}:
        return "quads"
    else:
        return "mixed"

def analyze(mesh, path):
    original_face_count, face_types = get_face_types(path)
    return {
        "Class": path.parent.name,
        "File": path.name,
        "Vertices Count": len(mesh.vertices),
        "Faces Count": original_face_count,
        "Face Type": classify(face_types),
        "Bounds Min": mesh.bounds[0].tolist(),
        "Bounds Max": mesh.bounds[1].tolist()
    }

# Step 2.5: Normalize the mesh
def normalize_mesh(mesh):
   # Center using barycenter
    barycenter = mesh.vertices.mean(axis=0)
    mesh.apply_translation(-barycenter)
       
   # Normalize size to largest dimension = 1
    max_extent = np.max(mesh.extents)

    if max_extent > 0:
        mesh.apply_scale(1.0 / max_extent)

    return mesh

# Step 2.5: Validate the normalization
def validate_normalization(mesh):
    # Check if the mesh is centered at the origin
    barycenter = mesh.vertices.mean(axis=0)
    max_extent = np.max(mesh.extents)

    print(f"Barycenter: {barycenter}, Max Extent: {max_extent}")

    # Official numpy documentation referenced
    print(f"Is centered: {np.allclose(barycenter, [0,0,0])}")
    print(f"Is normalized size: {np.isclose(max_extent, 1.0)}")

def plot_statistics(df):
    plt.figure()
    df["Vertices Count"].hist()
    plt.title("Vertices Count Distribution")
    plt.xlabel("Number of vertices")
    plt.ylabel("Number of shapes")
    plt.show()

    plt.figure()
    df["Faces Count"].hist()
    plt.title("Faces Count Distribution")
    plt.xlabel("Number of faces")
    plt.ylabel("Number of shapes")
    plt.show()

    plt.figure()
    df["Class"].value_counts().plot(kind='bar')
    plt.title("Shapes per class")
    plt.xlabel("Class")
    plt.ylabel("Number of shapes")
    plt.show()

# Function to find representative shapes based on vertices and faces count
def find_representative_shapes(df):
    # Find mean vertices and faces count
    mean_vertices = df["Vertices Count"].mean()
    mean_faces = df["Faces Count"].mean()

    # Find shape closest to the average
    df["Distance From Average"] = (abs(df["Vertices Count"] - mean_vertices) / mean_vertices
                                   + abs(df["Faces Count"] - mean_faces) / mean_faces)

    average_shape = df.loc[df["Distance From Average"].idxmin()]

    # Find some extreme shapes
    fewest_vertices = df.loc[df["Vertices Count"].idxmin()]
    most_vertices = df.loc[df["Vertices Count"].idxmax()]

    fewest_faces = df.loc[df["Faces Count"].idxmin()]
    most_faces = df.loc[df["Faces Count"].idxmax()]

    return average_shape, fewest_vertices, most_vertices, fewest_faces, most_faces



# Step 2.1 analysis
results = []
 
for path in sample_files:
    mesh = tr.load(path, force='mesh', process=False)
    results.append(analyze(mesh, path))

# Create and print dataframe
df = pd.DataFrame(results)
print(df)
df.to_csv("mesh_statistics.csv", index=False)

# Find the outliers in the dataset based on vertices and faces count
def find_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)

    IQR = Q3 - Q1

    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]

    return outliers

vertex_outliers = find_outliers(df, "Vertices Count")
face_outliers = find_outliers(df, "Faces Count")

print("Vertex outliers:")
print(
    vertex_outliers[
        ["Class", "File", "Vertices Count", "Faces Count"]
    ]
)

print("Face outliers:")
print(
    face_outliers[
        ["Class", "File", "Vertices Count", "Faces Count"]
    ]
)

# Lowest-resolution mesh
low_detail = df.loc[df["Vertices Count"].idxmin()]

# Highest-resolution mesh
high_detail = df.loc[df["Vertices Count"].idxmax()]

# Highest face-count mesh
high_faces = df.loc[df["Faces Count"].idxmax()]

# Step 2.2
print("Mean Vertices Count:", df["Vertices Count"].mean())
print("Mean Faces Count:", df["Faces Count"].mean())

plot_statistics(df)

average_shape, fewest_vertices, most_vertices, fewest_faces, most_faces = find_representative_shapes(df)

representative_shapes = [
    ("Average-like shape", average_shape),
    ("Fewest vertices", fewest_vertices),
    ("Most vertices", most_vertices),
    ("Fewest faces", fewest_faces),
    ("Most faces", most_faces),
]

shapes_to_view = [("Average-like shape", average_shape), ("Fewest vertices", fewest_vertices), ("Most vertices", most_vertices), ("Low-detail outlier", low_detail), ("High-detail outlier", high_detail), ("High face-count outlier", high_faces)]

for label, row in shapes_to_view:
    print(
        f"{label}: {row['File']} | "
        f"Vertices: {row['Vertices Count']} | "
        f"Faces: {row['Faces Count']}"
    )


for label,row in shapes_to_view:
    path = data_path / row["Class"] / row["File"]

    mesh = tr.load(path, force='mesh', process=False)
    show_mesh(mesh, path)
