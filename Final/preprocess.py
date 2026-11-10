"""Repair, resample, align, and normalize source meshes into an OBJ database."""

import argparse
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import trimesh as tr
from tqdm import tqdm

import MMR_utility as mmru


MIN_VERTICES = 5000
MAX_VERTICES = 10000
MAX_RESAMPLING_ITERATIONS = 15
PROJECT_DIR = Path(__file__).resolve().parent


# Mesh orientation, alignment, and scale normalization

def _orientation_sign(values):
    """Choose a deterministic sign for one PCA axis projection."""
    values = np.asarray(values, dtype=np.float64)
    score = float(np.sum(np.sign(values) * values**2))
    if np.isclose(score, 0.0) and len(values):
        score = float(values[int(np.argmax(np.abs(values)))])
    return -1.0 if score < 0 else 1.0


def center_mesh(mesh):
    """Translate a mesh so its vertex barycenter sits at the origin.

    Args:
        mesh: Mesh to center; a copy is translated.
    Returns:
        Centered copy of the mesh.
    """
    mesh = mesh.copy()
    mesh.apply_translation(-mesh.vertices.mean(axis=0))
    return mesh


def align_to_principal_axes(mesh):
    """Rotate a centered mesh so its PCA axes map to x, y, and z.

    Args:
        mesh: Centered Trimesh to align.
    Returns:
        New Trimesh with the largest-variance axis on x, then y, then z.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    covariance = np.cov(vertices, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    axes = eigenvectors[:, np.argsort(eigenvalues)[::-1]]
    for axis_index in (0, 1):
        pivot = int(np.argmax(np.abs(axes[:, axis_index])))
        if axes[pivot, axis_index] < 0:
            axes[:, axis_index] *= -1
    axes[:, 2] = np.cross(axes[:, 0], axes[:, 1])
    return tr.Trimesh(vertices @ axes, mesh.faces.copy(), process=False)


def flip_axes(mesh):
    """Mirror each axis so the heavier side of the mesh points toward +x/+y/+z.

    Args:
        mesh: PCA-aligned Trimesh.
    Returns:
        New Trimesh with a deterministic orientation sign on every axis.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    signs = np.array([
        _orientation_sign(vertices[:, axis]) for axis in range(3)
    ])
    return tr.Trimesh(vertices * signs, mesh.faces.copy(), process=False)


def mesh_alignment(mesh):
    """Align a mesh to principal axes and apply deterministic axis flips.

    Args:
        mesh: Centered Trimesh to align.
    Returns:
        New Trimesh with PCA axes mapped to x, y, and z.
    """
    return flip_axes(align_to_principal_axes(mesh))


def scale_to_unit(mesh):
    """Uniformly scale a mesh so its largest axis extent equals one.

    Args:
        mesh: Mesh to scale; a copy is scaled.
    Returns:
        Scaled copy of the mesh.
    """
    mesh = mesh.copy()
    maximum_extent = float(np.max(mesh.extents))
    if not np.isfinite(maximum_extent) or maximum_extent <= 0:
        raise ValueError("Cannot normalize a mesh with zero or invalid extent")
    mesh.apply_scale(1.0 / maximum_extent)
    return mesh


def normalize_mesh(mesh):
    """Center, PCA-align, flip, and scale a mesh to unit maximum extent.

    Args:
        mesh: Mesh to normalize; a copy is transformed.
    Returns:
        Centered, aligned mesh with largest axis extent equal to one.
    """
    return scale_to_unit(mesh_alignment(center_mesh(mesh)))


# Repair and vertex-count adjustment

def repair_mesh(mesh):
    """Merge duplicate vertices, remove duplicate faces, and fill mesh holes.

    Args:
        mesh: Input mesh to repair.
    Returns:
        Tuple of repaired mesh, change flag, and removed component count.
    """
    working_mesh = mesh.copy()
    vertex_count = len(working_mesh.vertices)
    face_count = len(working_mesh.faces)
    working_mesh.merge_vertices()
    working_mesh.update_faces(working_mesh.unique_faces())
    working_mesh.remove_unreferenced_vertices()
    if not working_mesh.is_watertight:
        working_mesh.fill_holes()
    changed = (
        len(working_mesh.vertices) != vertex_count
        or len(working_mesh.faces) != face_count
    )
    if not len(working_mesh.vertices) or not len(working_mesh.faces):
        raise ValueError("Repair removed all mesh geometry")
    return working_mesh, changed, 0


def refine_to_vertex_range(mesh, minimum, maximum, target, max_iterations):
    """Subdivide a mesh until its vertex count enters the requested range.

    Args:
        mesh: Mesh to refine.
        minimum: Inclusive minimum vertex count.
        maximum: Inclusive maximum vertex count.
        target: Preferred vertex count within the range.
        max_iterations: Maximum subdivision attempts.
    Returns:
        Tuple of best mesh, iterations used, and whether it reached the range.
    """
    best_mesh = mesh.copy()
    best_distance = abs(len(best_mesh.vertices) - target)
    edge_size = float(best_mesh.edges_unique_length.mean())
    if not np.isfinite(edge_size) or edge_size <= 0:
        raise ValueError("Mesh has no positive finite mean edge length")
    low_edge = high_edge = None
    for iteration in range(max_iterations):
        candidate = mesh.copy().subdivide_to_size(max_edge=edge_size)
        count = len(candidate.vertices)
        if abs(count - target) < best_distance:
            best_mesh = candidate
            best_distance = abs(count - target)
        if minimum <= count <= maximum:
            return candidate, iteration + 1, True
        if count < target:
            high_edge = edge_size
        else:
            low_edge = edge_size
        if low_edge is not None and high_edge is not None:
            edge_size = (low_edge + high_edge) / 2.0
        else:
            edge_size *= np.sqrt(count / target)
    return best_mesh, max_iterations, minimum <= len(best_mesh.vertices) <= maximum


def decimate_to_vertex_range(mesh, minimum, maximum, target, max_iterations):
    """Simplify a mesh until its vertex count enters the requested range.

    Args:
        mesh: Mesh to simplify.
        minimum: Inclusive minimum vertex count.
        maximum: Inclusive maximum vertex count.
        target: Preferred vertex count within the range.
        max_iterations: Maximum quadric-decimation attempts.
    Returns:
        Tuple of best mesh, iterations used, and whether it reached the range.
    """
    best_mesh = mesh.copy()
    best_distance = abs(len(best_mesh.vertices) - target)
    original_vertices = len(mesh.vertices)
    original_faces = len(mesh.faces)
    if original_vertices <= 0 or original_faces <= 0:
        raise ValueError("Mesh has no vertices or faces to simplify")
    face_count = max(1, round(target * original_faces / original_vertices))
    iterations = 0
    for iteration in range(max_iterations):
        candidate = mesh.copy().simplify_quadric_decimation(face_count=face_count)
        count = len(candidate.vertices)
        iterations = iteration + 1
        if abs(count - target) < best_distance:
            best_mesh = candidate
            best_distance = abs(count - target)
        if minimum <= count <= maximum:
            return candidate, iterations, True
        next_face_count = max(1, round(face_count * target / max(count, 1)))
        if next_face_count == face_count:
            break
        face_count = next_face_count
    return best_mesh, iterations, minimum <= len(best_mesh.vertices) <= maximum


def resample_mesh(mesh, minimum_vertices=MIN_VERTICES,
                  maximum_vertices=MAX_VERTICES,
                  max_iterations=MAX_RESAMPLING_ITERATIONS):
    """Refine or decimate a mesh into the vertex range.

    Returns:
        Tuple of mesh, method name, iteration count, and in-range flag. 
    """
    target = (minimum_vertices + maximum_vertices) // 2
    count = len(mesh.vertices)
    method, iterations = "unchanged", 0
    if count < minimum_vertices:
        mesh, iterations, converged = refine_to_vertex_range(
            mesh, minimum_vertices, maximum_vertices, target, max_iterations
        )
        method = "refined"
        if len(mesh.vertices) > maximum_vertices:
            mesh, correction_iterations, converged = decimate_to_vertex_range(
                mesh, minimum_vertices, maximum_vertices, maximum_vertices,
                max_iterations,
            )
            iterations += correction_iterations
            method = "refined+corrected"
    elif count > maximum_vertices:
        mesh, iterations, converged = decimate_to_vertex_range(
            mesh, minimum_vertices, maximum_vertices, maximum_vertices,
            max_iterations,
        )
        method = "simplified"
    else:
        converged = True

    return mesh, method, iterations, converged


# One canonical preprocessing sequence, shared by database and query runs

def preprocess_mesh(mesh, minimum_vertices=MIN_VERTICES,
                    maximum_vertices=MAX_VERTICES,
                    max_iterations=MAX_RESAMPLING_ITERATIONS):
    """Repair, resample, center, align, and normalize one mesh in that order.

    Args:
        mesh: Raw loaded mesh.
        minimum_vertices: Inclusive minimum final vertex count.
        maximum_vertices: Inclusive maximum final vertex count.
        max_iterations: Maximum subdivision or decimation iterations.
    Returns:
        Tuple of normalized mesh and preprocessing metadata.
    """
    original_vertices, original_faces = len(mesh.vertices), len(mesh.faces)
    mesh, repaired, removed_components = repair_mesh(mesh)
    mesh, method, iterations, converged = resample_mesh(
        mesh, minimum_vertices, maximum_vertices, max_iterations
    )

    mesh = normalize_mesh(mesh)
    metadata = {
        "Original_vertices": original_vertices,
        "Original_faces": original_faces,
        "Final_vertices": len(mesh.vertices),
        "Final_faces": len(mesh.faces),
        "Was_repaired": repaired,
        "Components_removed": removed_components,
        "Method": method,
        "Iterations": iterations,
        "Converged": converged,
    }
    return mesh, metadata


def preprocess_database(source_path, output_path, samples_per_category=None,
                        seed=43, resume=False):
    """Preprocess a database or stratified subset and write meshes plus a log.

    Args:
        source_path: Original database root.
        output_path: Destination root for processed meshes and a CSV log.
        samples_per_category: Optional maximum selection per category.
        seed: Seed for reproducible stratified selection.
        resume: Reuse successful output meshes and retry missing/failed entries.
    Returns:
        DataFrame with one preprocessing status row per selected mesh.
    """
    source_path = Path(source_path).resolve()
    output_path = Path(output_path).resolve()
    paths = mmru.find_obj_files(source_path)
    if samples_per_category is not None:
        paths = mmru.stratified_mesh_sample(paths, samples_per_category, seed)
    if not paths:
        raise ValueError(f"No OBJ meshes found under {source_path}")
    has_existing_meshes = output_path.exists() and any(output_path.rglob("*.obj"))
    if has_existing_meshes and not resume:
        raise FileExistsError(
            f"Processed meshes already exist under {output_path}; "
            "choose a fresh output directory or pass resume=True"
        )
    output_path.mkdir(parents=True, exist_ok=True)
    log_path = output_path / "preprocessing_log.csv"
    previous_rows = {}
    if resume and log_path.exists():
        previous = pd.read_csv(log_path).fillna("")
        previous_rows = {
            (row["Class"], row["File"]): row.to_dict()
            for _, row in previous.iterrows()
        }

    rows = []
    for mesh_path in tqdm(paths, desc="Preprocessing meshes"):
        relative = mesh_path.relative_to(source_path)
        key = (relative.parent.name, relative.name)
        destination = output_path / relative
        old_row = previous_rows.get(key)
        if (
            resume
            and old_row is not None
            and old_row.get("Status") in {"OK", "WARNING"}
            and destination.is_file()
        ):
            rows.append(old_row)
            continue
        row = {
            "Class": relative.parent.name,
            "File": relative.name,
            "Status": "ERROR",
            "Error": "",
            "Warning": "",
        }
        try:
            mesh, metadata = preprocess_mesh(mmru.load_mesh(mesh_path))
            destination.parent.mkdir(parents=True, exist_ok=True)
            mesh.export(destination, file_type="obj")
            row.update(metadata)
            row["Status"] = "OK" if metadata["Converged"] else "WARNING"
            if not metadata["Converged"]:
                row["Warning"] = (
                    f"Best effort retained at {metadata['Final_vertices']} vertices; "
                    f"target range is [{MIN_VERTICES}, {MAX_VERTICES}]"
                )
        except Exception as error:
            row["Error"] = str(error)
        rows.append(row)

    results = pd.DataFrame(rows)
    results.to_csv(log_path, index=False)
    return results


def preprocess_query_mesh(mesh_path, seed_root=None):
    """Preprocess and OBJ-round-trip one query using the database code path.

    Args:
        mesh_path: Raw query OBJ path.
        seed_root: Optional root for deriving the stable query identifier.
    Returns:
        Tuple of canonical reloaded query mesh, preprocessing metadata, and ID.
    """
    mesh_path = Path(mesh_path)
    mesh, metadata = preprocess_mesh(mmru.load_mesh(mesh_path))
    with tempfile.TemporaryDirectory(prefix="mmr-query-") as temporary_dir:
        temporary_path = Path(temporary_dir) / mesh_path.name
        mesh.export(temporary_path, file_type="obj")
        canonical_mesh = mmru.load_mesh(temporary_path)
    return canonical_mesh, metadata, mmru.mesh_identifier(mesh_path, seed_root)


def main():
    """Parse options, run the chosen database preprocessing job, and summarize."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--full", action="store_true",
        help="Process the entire source database instead of a stratified sample.",
    )
    parser.add_argument(
        "--sample-per-category", type=int, default=10,
        help="Meshes selected per category in sample mode (default: 10).",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--resume", action="store_true",
        help="Reuse successful outputs and retry failed or missing meshes.",
    )
    parser.add_argument(
        "--source", type=Path,
        default=PROJECT_DIR / "shapes" / "ShapeDatabase_INFOMR",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.sample_per_category < 1:
        parser.error("--sample-per-category must be at least one")

    sample_count = None if args.full else args.sample_per_category
    default_output = PROJECT_DIR / "shapes" / (
        "ShapeDatabase_INFOMR_cleaned" if args.full
        else "ShapeDatabase_INFOMR_cleaned_sample"
    )
    output_path = args.output or default_output
    results = preprocess_database(
        args.source,
        output_path,
        samples_per_category=sample_count,
        seed=args.seed,
        resume=args.resume,
    )
    successful = int(results["Status"].isin(["OK", "WARNING"]).sum())
    warnings = int((results["Status"] == "WARNING").sum())
    failed = int((results["Status"] == "ERROR").sum())
    print(f"Meshes selected: {len(results)}")
    print(f"Preprocessed successfully: {successful}")
    print(f"Vertex-range warnings: {warnings}")
    print(f"Failed: {failed}")
    print(f"Database: {output_path}")
    print(f"Log: {output_path / 'preprocessing_log.csv'}")


if __name__ == "__main__":
    main()
