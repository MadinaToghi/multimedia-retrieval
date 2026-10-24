import trimesh as tr
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt
import csv
from pathlib import Path
from vedo import Mesh, show
from tqdm import tqdm

MIN_VERTICES = 5000
MAX_VERTICES = 10000
MAX_ITERATIONS = 15

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
        new=True
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

def mesh_orientation(x, y, z):
    x_orient = np.sign(sum(np.sign(item) * np.square(item) for item in x))
    y_orient = np.sign(sum(np.sign(item) * np.square(item) for item in y))
    z_orient = np.sign(sum(np.sign(item) * np.square(item) for item in z))

    return np.diag([x_orient, y_orient, z_orient, 1.0])

def find_mesh_eigenvectors(mesh):
     
    A_cov = np.cov(np.array(mesh.vertices).T)  # 3x3 matrix
    # computes the eigenvalues and eigenvectors for the covariance matrix. 
    eigenvalues, eigenvectors = np.linalg.eig(A_cov)
    order = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, order]

    return eigenvectors

def mesh_alignment(mesh):
    #Step 3.1.1: align
    
    eigenvectors = find_mesh_eigenvectors(mesh)
   
    pc1 = eigenvectors[:,0]
    pc2 = eigenvectors[:,1]

    x_updated = np.dot(mesh.vertices, pc1)
    y_updated = np.dot(mesh.vertices, pc2)
    z_updated = np.dot(mesh.vertices, np.cross(pc1, pc2))

    new_vertices = np.array([x_updated, y_updated, z_updated]).T

    new_mesh = tr.Trimesh(new_vertices, mesh.faces, process=False)

    rotation = mesh_orientation(x_updated, y_updated, z_updated)
    new_mesh.apply_transform(rotation)

    return new_mesh

# Step 2.5: Normalize the mesh
def normalize_mesh(mesh):
   # Center using barycenter
    barycenter = mesh.vertices.mean(axis=0)
    mesh.apply_translation(-barycenter)

    mesh = mesh_alignment(mesh)
    
       
   # Normalize size to largest dimension = 1
    max_extent = np.max(mesh.extents)

    if max_extent > 0:
        mesh.apply_scale(1.0 / max_extent)

    return mesh

def repair_mesh(mesh):
    """Fill holes and merge vertices; return the mesh, repair flag, and removed-component count."""
    original_mesh = mesh
    components_removed = 0

    try:
        working_mesh = mesh.copy()

        vertices_before = len(working_mesh.vertices)
        faces_before = len(working_mesh.faces)

        # Merge duplicate vertices
        working_mesh.merge_vertices()

        # Remove duplicate faces
        working_mesh.update_faces(
            working_mesh.unique_faces()
        )

        # Remove unused vertices
        working_mesh.remove_unreferenced_vertices()
    
        # Attempt to fill holes
        if not working_mesh.is_watertight:
            working_mesh.fill_holes()

        # Check whether geometry changed
        was_repaired = (
            len(working_mesh.vertices) != vertices_before
            or len(working_mesh.faces) != faces_before
        )

        return working_mesh, was_repaired, components_removed

    except Exception as error:
        original_mesh.metadata.setdefault(
            "_normalization_errors", []
        ).append(str(error))

        return original_mesh, False, 0


def refine_to_range(mesh, min_v, max_v, target_v, max_iterations):
    """Subdivide a mesh toward the vertex range; return the best mesh, iterations, and convergence flag."""
    best_mesh = mesh
    iterations_used = 0
    low_edge = None
    high_edge = None

    try:
        best_distance = abs(len(mesh.vertices) - target_v)
        if min_v <= len(mesh.vertices) <= max_v:
            return best_mesh, iterations_used, True

        original_mesh = mesh.copy()
        max_edge = float(original_mesh.edges_unique_length.mean())
        if not np.isfinite(max_edge) or max_edge <= 0:
            raise ValueError("Mesh has no positive finite mean edge length")

        for iteration in range(max_iterations):
            candidate = original_mesh.copy().subdivide_to_size(max_edge=max_edge)
            candidate_count = len(candidate.vertices)
            iterations_used = iteration + 1
            distance = abs(candidate_count - target_v)
            if distance < best_distance:
                best_mesh = candidate
                best_distance = distance

            if min_v <= candidate_count <= max_v:
                return candidate, iterations_used, True

            if candidate_count < target_v:
                high_edge = max_edge
            else:
                low_edge = max_edge

            if low_edge is not None and high_edge is not None:
                max_edge = (low_edge + high_edge) / 2.0
            else:
                max_edge *= np.sqrt(candidate_count / target_v)
                if not np.isfinite(max_edge) or max_edge <= 0:
                    break
    except Exception as error:
        best_mesh.metadata.setdefault("_normalization_errors", []).append(str(error))

    return best_mesh, iterations_used, min_v <= len(best_mesh.vertices) <= max_v


def decimate_to_range(mesh, min_v, max_v, target_v, max_iterations):
    """Simplify a mesh toward the vertex range; return the best mesh, iterations, and convergence flag."""
    best_mesh = mesh
    iterations_used = 0

    try:
        best_distance = abs(len(mesh.vertices) - target_v)
        if min_v <= len(mesh.vertices) <= max_v:
            return best_mesh, iterations_used, True

        original_mesh = mesh.copy()
        original_vertex_count = len(original_mesh.vertices)
        original_face_count = len(original_mesh.faces)
        if original_vertex_count <= 0 or original_face_count <= 0:
            raise ValueError("Mesh has no vertices or faces to simplify")

        face_count = max(1, round(target_v * original_face_count / original_vertex_count))
        for iteration in range(max_iterations):
            candidate = original_mesh.copy().simplify_quadric_decimation(face_count=face_count)
            candidate_count = len(candidate.vertices)
            iterations_used = iteration + 1
            distance = abs(candidate_count - target_v)
            if distance < best_distance:
                best_mesh = candidate
                best_distance = distance

            if min_v <= candidate_count <= max_v:
                return candidate, iterations_used, True

            next_face_count = max(1, round(face_count * target_v / max(candidate_count, 1)))
            if next_face_count == face_count:
                break
            face_count = next_face_count
    except Exception as error:
        best_mesh.metadata.setdefault("_normalization_errors", []).append(str(error))

    return best_mesh, iterations_used, min_v <= len(best_mesh.vertices) <= max_v


def process_single_mesh(path, data_path, output_root):
    """Normalize, repair, range-adjust, and export one mesh; return its CSV row as a dictionary."""
    relative_path = Path(path).name
    row = {
        "filename": relative_path,
        "original_vertex_count": "",
        "post_normalize_vertex_count": "",
        "was_repaired": False,
        "components_removed": 0,
        "final_vertex_count": "",
        "method_used": "unchanged",
        "iterations_used": 0,
        "converged": False,
        "status": "ERROR",
        "error": "",
    }

    try:
        path = Path(path)
        data_path = Path(data_path)
        output_root = Path(output_root)
        relative_path = path.resolve().relative_to(data_path.resolve())
        row["filename"] = str(relative_path)

        mesh = tr.load(path, force="mesh", process=False)
        row["original_vertex_count"] = len(mesh.vertices)

        # === Original spot for normalization, didn't work with alignment ===
        # mesh = normalize_mesh(mesh) 
        # row["post_normalize_vertex_count"] = len(mesh.vertices)
        # ===================================================================

        mesh, was_repaired, components_removed = repair_mesh(mesh)
        row["was_repaired"] = was_repaired
        row["components_removed"] = components_removed
        errors = list(mesh.metadata.pop("_normalization_errors", []))

       

        vertex_count = len(mesh.vertices)
        if MIN_VERTICES <= vertex_count <= MAX_VERTICES:
            method_used = "unchanged"
        elif vertex_count < MIN_VERTICES:
            mesh, iterations_used, _ = refine_to_range(
                mesh, MIN_VERTICES, MAX_VERTICES,
                (MIN_VERTICES + MAX_VERTICES) // 2, MAX_ITERATIONS,
            )
            method_used = "refined"
            errors.extend(mesh.metadata.pop("_normalization_errors", []))
            if len(mesh.vertices) > MAX_VERTICES:
                mesh, correction_iterations, _ = decimate_to_range(
                    mesh, MIN_VERTICES, MAX_VERTICES,
                    MAX_VERTICES, MAX_ITERATIONS,
                )
                iterations_used += correction_iterations
                method_used = "refined+corrected"
                errors.extend(mesh.metadata.pop("_normalization_errors", []))
        else:
            mesh, iterations_used, _ = decimate_to_range(
                mesh, MIN_VERTICES, MAX_VERTICES,
                MAX_VERTICES, MAX_ITERATIONS,
            )
            method_used = "simplified"
            errors.extend(mesh.metadata.pop("_normalization_errors", []))

        # ===== Normalization works when it is put here! =====
        mesh = normalize_mesh(mesh) 
        row["post_normalize_vertex_count"] = len(mesh.vertices)

        # Validate final mesh
        duplicate_faces = len(mesh.faces) - int(mesh.unique_faces().sum())

        if duplicate_faces > 0:
            print(f"WARNING: {path.name} still has {duplicate_faces} duplicate faces")
        # ====================================================

        output_path = output_root / relative_path.with_suffix(".obj")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        mesh.export(output_path, file_type="obj")

        final_vertex_count = len(mesh.vertices)
        row.update({
            "final_vertex_count": final_vertex_count,
            "method_used": method_used,
            "iterations_used": iterations_used if method_used != "unchanged" else 0,
            "converged": MIN_VERTICES <= final_vertex_count <= MAX_VERTICES,
            "status": "OK",
            "error": "; ".join(errors),
        })
    except Exception as error:
        row["error"] = str(error)

    return row


def normalize_sample_files(files, data_path, output_root):
    """Process a batch of meshes and write each result row to normalization_log.csv."""
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    columns = [
        "filename", "original_vertex_count", "post_normalize_vertex_count",
        "was_repaired", "components_removed", "final_vertex_count",
        "method_used", "iterations_used", "converged", "status", "error",
    ]

    with (output_root / "normalization_log.csv").open("w", newline="", encoding="utf-8") as log_file:
        writer = csv.DictWriter(log_file, fieldnames=columns)
        writer.writeheader()
        for path in tqdm(files):
            row = process_single_mesh(path, data_path, output_root)
            writer.writerow(row)
            log_file.flush()

# Step 2.5: Validate the normalization
def validate_normalization(mesh):
    # Check if the mesh is centered at the origin
    barycenter = mesh.vertices.mean(axis=0)
    max_extent = np.max(mesh.extents)
    mesh_eigenvectors = find_mesh_eigenvectors(mesh)
    mesh_eigenvectors = np.round(mesh_eigenvectors.real, 2)

    print(f"Barycenter: {barycenter}, Max Extent: {max_extent}")
    print(f"eigenvectors:")
    print(mesh_eigenvectors)

    # Official numpy documentation referenced
    print(f"Is centered: {np.allclose(barycenter, [0,0,0])}")
    print(f"Is aligned: {np.allclose(np.abs(mesh_eigenvectors), np.identity(3))}")
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


def plot_normalization_results(log_path):
    """Display vertex-count, method-count, and before/after plots from the batch CSV."""
    results = pd.read_csv(log_path)
    successful = results[results["status"] == "OK"].copy()
    for column in ("post_normalize_vertex_count", "final_vertex_count"):
        successful[column] = pd.to_numeric(successful[column], errors="coerce")
    successful = successful.dropna(subset=["post_normalize_vertex_count", "final_vertex_count"])

    maximum_vertices = max(
        MAX_VERTICES,
        int(np.ceil(successful["final_vertex_count"].max() / 500) * 500),
    )
    bins = np.arange(0, maximum_vertices + 501, 500)
    figure, axis = plt.subplots()
    axis.hist(successful["final_vertex_count"], bins=bins, color="steelblue", edgecolor="white")
    axis.axvline(MIN_VERTICES, color="darkorange", linestyle="--", label=f"Minimum ({MIN_VERTICES:,})")
    axis.axvline(MAX_VERTICES, color="firebrick", linestyle="--", label=f"Maximum ({MAX_VERTICES:,})")
    axis.set(title="Final Vertex Counts", xlabel="Vertex count", ylabel="Number of OBJ files")
    axis.legend()
    figure.tight_layout()

    method_order = ["refined", "simplified", "unchanged"]
    method_labels = ["Refined", "Decimated", "Unchanged"]
    method_counts = successful["method_used"].value_counts()
    figure, axis = plt.subplots()
    axis.bar(method_labels, [method_counts.get(method, 0) for method in method_order], color=[
        "seagreen", "indianred", "slategray",
    ])
    axis.set(title="Objects by Processing Method", ylabel="Number of OBJ files")
    axis.tick_params(axis="x", rotation=15)
    figure.tight_layout()

    figure, axis = plt.subplots()
    method_styles = {
        "refined": ("seagreen", "Refined"),
        "refined+corrected": ("mediumseagreen", "Refined + corrected"),
        "simplified": ("indianred", "Decimated"),
    }
    changed = successful[successful["method_used"].isin(method_styles)]
    for method, (color, label) in method_styles.items():
        group = changed[changed["method_used"] == method]
        if not group.empty:
            axis.scatter(
                group["post_normalize_vertex_count"],
                group["final_vertex_count"],
                s=24,
                alpha=0.65,
                color=color,
                label=label,
            )

    plot_minimum = max(1, int(min(
        changed["post_normalize_vertex_count"].min(),
        changed["final_vertex_count"].min(),
    )))
    plot_maximum = int(max(
        changed["post_normalize_vertex_count"].max(),
        changed["final_vertex_count"].max(),
    ))
    axis.plot([plot_minimum, plot_maximum], [plot_minimum, plot_maximum], color="black", linestyle=":", label="No count change")
    for boundary in (MIN_VERTICES, MAX_VERTICES):
        axis.axvline(boundary, color="gray", linestyle="--", alpha=0.55)
        axis.axhline(boundary, color="gray", linestyle="--", alpha=0.55)
    axis.set_xscale("log")
    axis.set_yscale("log")
    axis.set(
        title="Vertex Counts Before and After Range Adjustment",
        xlabel="Vertices after normalization (before repair)",
        ylabel="Final vertices",
    )
    axis.legend()
    figure.tight_layout()
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
"""
results = []
 
for path in sample_files:
    mesh = tr.load(path, force='mesh', process=False)
    results.append(analyze(mesh, path))

# Create and print dataframe
df = pd.DataFrame(results)
print(df)
df.to_csv("mesh_statistics.csv", index=False)

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

for label, row in representative_shapes:
    print(
        f"{label}: {row['File']} | "
        f"Vertices: {row['Vertices Count']} | "
        f"Faces: {row['Faces Count']}"
    )


for label,row in representative_shapes:
    path = data_path / row["Class"] / row["File"]

    mesh = tr.load(path, force='mesh', process=False)
    show_mesh(mesh, path)
"""

output_root = Path(
    "../ShapeDatabase_INFOMR_normalized_cleaned"
)
normalize_sample_files(sample_files, data_path, output_root)
plot_normalization_results(output_root / "normalization_log.csv")

# shown_categories = set()
# for sample_path in sample_files:
#     category = sample_path.parent.name
#     if category in shown_categories:
#         continue

#     normalized_path = output_root / sample_path.relative_to(data_path).with_suffix(".obj")
#     original_mesh = tr.load(sample_path, force="mesh", process=False)
#     normalized_mesh = tr.load(normalized_path, force="mesh", process=False)
#     print(
#         f"{category}/{sample_path.name} | "
#         f"original_vertices={len(original_mesh.vertices)}, "
#         f"normalized_vertices={len(normalized_mesh.vertices)}"
#     )
#     print(f"Showing original | {category}/{sample_path.name}")
#     validate_normalization(original_mesh)
#     show_mesh(original_mesh, sample_path.with_name(f"{sample_path.stem}__original.obj"))
    
    
#     print(f"Showing normalized | {category}/{normalized_path.name}")
#     validate_normalization(normalized_mesh)
#     show_mesh(normalized_mesh, normalized_path.with_name(f"{normalized_path.stem}__normalized.obj"))
    
#     shown_categories.add(category)
#     if len(shown_categories) == 10:
#         break
