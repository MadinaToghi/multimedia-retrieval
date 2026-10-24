import trimesh as tr
import numpy as np
import pandas as pd
from vedo import Mesh, show

from pathlib import Path

# Aan het einde: output of 3.1
from pathlib import Path

# Original database
original_path = Path(
    "../ShapeDatabase_INFOMR"
)

# Normalized database
normalized_path = Path(
    "../ShapeDatabase_INFOMR_normalized_cleaned"
)

# Keep this for your global descriptor calculations
data_path = normalized_path

obj_files = list(data_path.rglob("*.obj"))

# Global Descriptors Calculation
def surface_area(mesh):
    # Total surface area of the mesh
    return mesh.area


def compactness(mesh):
    # Ensure watertight
    #if not mesh.is_watertight:
    #    print("Warning: Mesh is not watertight. Volume calculation may be inaccurate.")
    
    volume = abs(mesh.volume)
    area = mesh.area

    # Handle edge case for empty or flat meshes
    if area <= 0:
        return np.nan
    
    # Shpere has compactness = 1
    compactness = (36.0 * np.pi * (volume**2)) / (area**3)

    return compactness


def rectangularity(mesh):
    # Shape volume / volume of the oriented bounding box
    volume = abs(mesh.volume)
    obb_volume = mesh.bounding_box_oriented.volume

    if obb_volume <= 0:
        return np.nan
    
    rectangularity = volume / obb_volume

    return rectangularity


def diameter(mesh):
    # Diameter = max Euclidean distance between two points
    hull_vertices = mesh.convex_hull.vertices

    max_distance = 0.0

    for vertex in hull_vertices:
        distances = np.linalg.norm(hull_vertices - vertex, axis=1)
        max_distance = max(max_distance, distances.max())

    return max_distance


def convexity(mesh):
    # shape_volume / volume of the convex hull

    volume = abs(mesh.volume)
    hull_volume = mesh.convex_hull.volume

    if hull_volume <= 0:
        return np.nan
    
    convexity = volume / hull_volume

    return convexity


def eccentricity(mesh):
    # Compute the eccentricity of the mesh
    # Eccentricity is defined as the ratio of the distance between the foci of the ellipse and its major axis length
    covariance_matrix = np.cov(mesh.vertices, rowvar=False)

    eigenvalues = np.linalg.eigvalsh(covariance_matrix)

    smallest_eigenvalue = np.min(eigenvalues)
    largest_eigenvalue = np.max(eigenvalues)

    if smallest_eigenvalue <= 0 or largest_eigenvalue <= 0:
        return np.nan

    eccentricity = largest_eigenvalue / smallest_eigenvalue    

    return eccentricity

results = []
skipped_meshes = []

for path in obj_files:
    print(f"Processing: {path.name}")

    mesh = tr.load(path, force="mesh", process=False)

    volume = abs(mesh.volume)

    if not np.isfinite(volume) or np.isclose(volume, 0):
        skipped_meshes.append({"Class": path.parent.name, "File": path.name,
                               "Reason": "Invalid or near-zero volume"})
        continue

    results.append({
        "Class": path.parent.name,
        "File": path.name,
        "Volume": volume,
        "Surface Area": surface_area(mesh),
        "Compactness": compactness(mesh),
        "Rectangularity": rectangularity(mesh),
        "Diameter": diameter(mesh),
        "Convexity": convexity(mesh),
        "Eccentricity": eccentricity(mesh),
    })

df = pd.DataFrame(results)

df.to_csv("shape_descriptors_cleaned.csv", index=False)

skipped_df = pd.DataFrame(
    skipped_meshes,
    columns=["Class", "File", "Reason"])

skipped_df.to_csv("skipped_meshes_cleaned.csv", index=False)

print(f"Successfully processed: {len(results)}")
print(f"Skipped meshes: {len(skipped_meshes)}")


# Choose the object
object_name = "D00063.obj"

# Find the file in any class folder
mesh_path = next(data_path.rglob(object_name), None)

if mesh_path is not None:
    mesh = tr.load(mesh_path, force="mesh", process=False)

    # Convert Trimesh to Vedo
    vedo_mesh = Mesh([mesh.vertices, mesh.faces])
    vedo_mesh.c("lightblue")
    vedo_mesh.compute_normals()

    # Show mesh edges
    edges = vedo_mesh.clone().wireframe().c("black").lw(1)

    # Display mesh
    show(
        vedo_mesh,
        edges,
        title=mesh_path.name,
        axes=1,
        viewup="z",
        interactive=True,
        new=True
    )
else:
    print("Mesh not found!")


object_name = "D00063.obj"
mesh_path = next(data_path.rglob(object_name))

mesh = tr.load(mesh_path, force="mesh", process=False)

print("BEFORE:")
print("Watertight:", mesh.is_watertight)
print("Winding consistent:", mesh.is_winding_consistent)
print("Volume:", mesh.volume)
print("Rectangularity:",
      abs(mesh.volume) / mesh.bounding_box_oriented.volume)
print("Convexity:",
      abs(mesh.volume) / mesh.convex_hull.volume)

# Test repair without changing the original OBJ file
repaired_mesh = mesh.copy()
repaired_mesh.fix_normals(multibody=True)

print("\nAFTER:")
print("Watertight:", repaired_mesh.is_watertight)
print("Winding consistent:", repaired_mesh.is_winding_consistent)
print("Volume:", repaired_mesh.volume)
print("Rectangularity:",
      abs(repaired_mesh.volume) / repaired_mesh.bounding_box_oriented.volume)
print("Convexity:",
      abs(repaired_mesh.volume) / repaired_mesh.convex_hull.volume)


from pathlib import Path
import trimesh as tr

# Correct database paths
original_path = Path(
    "../ShapeDatabase_INFOMR"
)

normalized_path = Path(
    "../ShapeDatabase_INFOMR_normalized_cleaned"
)

object_name = "D00063.obj"

for label, folder in [
    ("ORIGINAL", original_path),
    ("NORMALIZED", normalized_path)
]:
    path = next(folder.rglob(object_name), None)

    if path is None:
        print(f"{label}: File not found in {folder}")
        continue

    mesh = tr.load(path, force="mesh", process=False)

    print(f"\n{label}")
    print("File location:", path)
    print("Watertight:", mesh.is_watertight)
    print("Winding consistent:", mesh.is_winding_consistent)
    print("Vertices:", len(mesh.vertices))
    print("Faces:", len(mesh.faces))
    print("Volume:", mesh.volume)
    print("OBB volume:", mesh.bounding_box_oriented.volume)
    print("Convex hull volume:", mesh.convex_hull.volume)

    print("Rectangularity:",
          abs(mesh.volume) / mesh.bounding_box_oriented.volume)

    print("Convexity:",
          abs(mesh.volume) / mesh.convex_hull.volume)



import pandas as pd
import numpy as np

# Load existing descriptor CSV
df = pd.read_csv("shape_descriptors_cleaned.csv")

# Descriptors that should be between 0 and 1
bounded_descriptors = [
    "Compactness",
    "Rectangularity",
    "Convexity"
]

# Find all meshes with values outside the theoretical range
invalid_mask = (
    (df[bounded_descriptors] < 0) |
    (df[bounded_descriptors] > 1.000001) |
    ~np.isfinite(df[bounded_descriptors])
).any(axis=1)

# Create dataframe containing problematic meshes
invalid_meshes = df[invalid_mask].copy()

# Record which descriptors are invalid
def get_invalid_reasons(row):
    reasons = []

    for descriptor in bounded_descriptors:
        value = row[descriptor]

        if not np.isfinite(value):
            reasons.append(f"{descriptor}: non-finite")
        elif value < 0 or value > 1.000001:
            reasons.append(f"{descriptor}: {value:.4f}")

    return "; ".join(reasons)

invalid_meshes["Reason"] = invalid_meshes.apply(
    get_invalid_reasons, axis=1
)

# Save problematic meshes
invalid_meshes.to_csv("invalid_descriptors_cleaned.csv", index=False)

print(f"Total meshes checked: {len(df)}")
print(f"Meshes with invalid descriptors: {len(invalid_meshes)}")
print("Saved to invalid_descriptors.csv")

duplicate_counts = []

for path in obj_files:
    mesh = tr.load(path, force="mesh", process=False)

    total = len(mesh.faces)
    unique = int(mesh.unique_faces().sum())

    duplicate_counts.append({
        "Class": path.parent.name,
        "File": path.name,
        "Duplicate Faces": total - unique,
        "Duplicate Percentage": (
            100 * (total - unique) / total if total else 0
        )
    })

duplicate_df = pd.DataFrame(duplicate_counts)

print("Total meshes:", len(duplicate_df))
print(
    "Meshes with duplicate faces:",
    (duplicate_df["Duplicate Faces"] > 0).sum()
)

duplicate_df.to_csv("all_duplicate_faces_cleaned.csv", index=False)