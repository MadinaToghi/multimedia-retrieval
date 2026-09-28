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
files_by_category = {}

for path in obj_files:
    files_by_category.setdefault(path.parent.name, []).append(path)


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

def mesh_alignment(mesh):
    #Step 3.1.1: align
    pc1, pc2, pc3 = mesh.principal_inertia_vectors

    print(mesh)
    x_updated = np.dot(mesh.vertices, pc1)
    y_updated = np.dot(mesh.vertices, pc2)
    z_updated = np.dot(mesh.vertices, np.cross(pc1, pc2))

    new_vertices = np.array([x_updated, y_updated, z_updated]).T
    print(new_vertices.shape)


    print(mesh.faces.shape)
    print(x_updated.shape)
    new_mesh = tr.Trimesh(new_vertices, mesh.faces)
    
    return new_mesh


def normalize_mesh(mesh):
   # Center using barycenter
    #barycenter= mesh.center_mass
    barycenter = mesh.vertices.mean(axis=0)
    
    mesh.apply_translation(-barycenter)

    # Mesh alignment (Step 3.1)
    mesh = mesh_alignment(mesh)
       
   # Normalize size to largest dimension = 1
    max_extent = np.max(mesh.extents)

    if max_extent > 0:
        mesh.apply_scale(1.0 / max_extent)  # Note to self - weights

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


sample_files = []
for category, files in files_by_category.items():
    sample_files.extend(random.sample(files, min(10, len(files))))



path = sample_files[200]

mesh = tr.load_mesh(path, force='mesh', process=False)





#"""
print("mesh")
validate_normalization(mesh)



normal_mesh = normalize_mesh(mesh)
print("normalised mesh")
validate_normalization(normal_mesh)
show_mesh(normal_mesh, path)
#"""

"""
Notes

eigenvectors = mesh.principal_inertia_vectors

xi updated = (pi - c) ⋅ e1
yi updated = (pi - c) ⋅ e2
zi updated = (pi - c) ⋅ (e1 x e2)


"""
