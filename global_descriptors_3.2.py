import trimesh as tr
import numpy as np
import pandas as pd

from pathlib import Path

# Aan het einde: output of 3.1
data_path = Path(
    r"C:\Users\madin\Documents\multimedia-retrieval\ShapeDatabase_INFOMR_normalized"
)

obj_files = list(data_path.rglob("*.obj"))


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

    eigenvalues, _ = np.linalg.eig(covariance_matrix)

    smallest_eigenvalue = np.min(eigenvalues)
    largest_eigenvalue = np.max(eigenvalues)

    if smallest_eigenvalue <= 0 or largest_eigenvalue <= 0:
        return np.nan

    eccentricity = largest_eigenvalue / smallest_eigenvalue    

    return eccentricity


results = []

for path in obj_files:
    mesh = tr.load(path, force="mesh", process=False)

    results.append({
        "Class": path.parent.name,
        "File": path.name,
        "Surface Area": surface_area(mesh),
        "Compactness": compactness(mesh),
        "Rectangularity": rectangularity(mesh),
        "Diameter": diameter(mesh),
        "Convexity": convexity(mesh),
        "Eccentricity": eccentricity(mesh),
    })

df = pd.DataFrame(results)

df.to_csv("shape_descriptors.csv", index=False)

print(df)