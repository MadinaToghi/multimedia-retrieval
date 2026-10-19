import trimesh as tr
import numpy as np
import random
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from vedo import Mesh, show

data_path = Path("../ShapeDatabase_normalized")

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
    mesh_eigenvectors = find_mesh_eigenvectors(mesh).real

    print(f"Barycenter: {barycenter}, Max Extent: {max_extent}")
    print(f"eigenvectors:")
    np.set_printoptions(formatter={'float': '{:.2e}'.format})
    print(mesh_eigenvectors)

    # Official numpy documentation referenced
    print(f"Is centered: {np.allclose(barycenter, [0,0,0])}")
    print(f"Is aligned: {np.allclose(abs(mesh_eigenvectors), np.identity(3))}")
    print(f"Is normalized size: {np.isclose(max_extent, 1.0)}")


sample_files = []
for category, files in files_by_category.items():
    sample_files.extend(random.sample(files, min(10, len(files))))


ar=random.sample(range(690), 10)

for i in ar:

    path = sample_files[i]
    mesh = tr.load_mesh(path, force='mesh', process=False)
    print("Mesh pre-alignment")
    print(path.name)
    validate_normalization(mesh)
    show_mesh(mesh, path)

    normal_mesh = normalize_mesh(mesh)

    print("Mesh post-alignment")
    validate_normalization(normal_mesh)
    show_mesh(normal_mesh, path)





