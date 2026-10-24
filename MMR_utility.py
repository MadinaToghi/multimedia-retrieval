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

def show_mesh(mesh, path, isSamewindow=True):
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
        new=isSamewindow
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


def process_single_mesh(path):
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

        final_vertex_count = len(mesh.vertices)
        row.update({
            "final_vertex_count": final_vertex_count,
            "method_used": method_used,
            "iterations_used": iterations_used if method_used != "unchanged" else 0,
            "converged": MIN_VERTICES <= final_vertex_count <= MAX_VERTICES,
            "status": "OK",
            "error": "; ".join(errors),
        })

        return row, mesh
    except Exception as error:
        row["error"] = str(error)
        mesh = None

        return row, mesh

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