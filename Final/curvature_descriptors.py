"""Estimate fixed-radius curvature histograms and record per-mesh quality."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from tqdm import tqdm

import MMR_utility as mmru


CURVATURE_BIN_COUNT = 25
CURVATURE_MAX = 100.0
CURVATURE_RADIUS = 0.05
CURVATURE_QUERY_SAMPLE_COUNT = 3000
CURVATURE_SUPPORT_SAMPLES_PER_AREA = 6000
CURVATURE_MIN_SUPPORT_SAMPLES = 12000
CURVATURE_MIN_NEIGHBORS = 12
CURVATURE_MAX_NEIGHBORS = 50
CURVATURE_MIN_FIT_FRACTION = 0.95
CURVATURE_MIN_FACE_AREA = 1e-14
CURVATURE_BIN_EDGES = np.linspace(
    0.0, np.log1p(CURVATURE_MAX), CURVATURE_BIN_COUNT + 1
)
CURVATURE_BIN_CENTERS = (
    CURVATURE_BIN_EDGES[:-1] + CURVATURE_BIN_EDGES[1:]
) / 2.0
CURVATURE_COLUMNS = [
    f"Curvature_bin_{index}" for index in range(CURVATURE_BIN_COUNT)
]
PROJECT_DIR = Path(__file__).resolve().parent


# Area sampling and local quadratic curvature fitting

def _sample_surface(triangles, areas, count, rng):
    """Draw points uniformly by area from a set of triangular patches."""
    face_indices = rng.choice(len(triangles), size=count, p=areas / areas.sum())
    coordinates = rng.random((count, 2))
    outside = coordinates.sum(axis=1) > 1.0
    coordinates[outside] = 1.0 - coordinates[outside]
    return (
        triangles[face_indices, 0]
        + coordinates[:, 0, None]
        * (triangles[face_indices, 1] - triangles[face_indices, 0])
        + coordinates[:, 1, None]
        * (triangles[face_indices, 2] - triangles[face_indices, 0])
    )


def _fit_mean_curvature(point, neighbors, radius):
    """Fit a weighted quadratic patch and estimate absolute mean curvature."""
    centered = neighbors - point
    covariance = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[1] <= np.finfo(np.float64).eps * radius**2:
        return None
    local_coordinates = centered @ eigenvectors[:, [2, 1, 0]]
    x, y, z = (local_coordinates[:, axis] / radius for axis in range(3))
    radial_distance_squared = x * x + y * y
    weights = np.sqrt(np.exp(-2.0 * radial_distance_squared))
    design = np.column_stack(
        (np.ones(len(x)), x, y, 0.5 * x * x, x * y, 0.5 * y * y)
    )
    coefficients, _, rank, _ = np.linalg.lstsq(
        design * weights[:, None], z * weights, rcond=None
    )
    if rank < 6:
        return None
    slope_x, slope_y = coefficients[1:3] / radius
    second_xx, second_xy, second_yy = coefficients[3:6] / radius
    mean_curvature = (
        (1.0 + slope_y**2) * second_xx
        - 2.0 * slope_x * slope_y * second_xy
        + (1.0 + slope_x**2) * second_yy
    ) / (2.0 * (1.0 + slope_x**2 + slope_y**2) ** 1.5)
    if not np.isfinite(mean_curvature):
        return None
    return abs(float(mean_curvature))


def curvature_single(mesh, seed=0, radius=CURVATURE_RADIUS,
                     query_sample_count=CURVATURE_QUERY_SAMPLE_COUNT):
    """Estimate one normalized mesh's log-binned mean-curvature distribution.

    Args:
        mesh: Unit-extent triangular Trimesh.
        seed: Stable seed for area and support sampling.
        radius: Fixed neighborhood radius in normalized mesh units.
        query_sample_count: Number of area-uniform curvature query points.
    Returns:
        Tuple of normalized histogram, reliable-area fraction, and fit fraction.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if not len(vertices) or not len(faces):
        raise ValueError("Curvature requires a non-empty triangular mesh")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Curvature requires triangular faces")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Mesh vertices must be finite")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Curvature radius must be positive and finite")
    maximum_extent = float(np.max(np.ptp(vertices, axis=0)))
    if not np.isclose(maximum_extent, 1.0, rtol=0.0, atol=1e-3):
        raise ValueError("Curvature requires a mesh normalized to unit extent")

    triangles = vertices[faces]
    face_areas = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    ) / 2.0
    valid_faces = np.isfinite(face_areas) & (face_areas > CURVATURE_MIN_FACE_AREA)
    if not np.any(valid_faces):
        raise ValueError("Mesh has no non-degenerate triangles")

    valid_face_indices = faces[valid_faces]
    edges = np.concatenate((
        valid_face_indices[:, [0, 1]],
        valid_face_indices[:, [1, 2]],
        valid_face_indices[:, [2, 0]],
    ))
    edges.sort(axis=1)
    unique_edges, edge_counts = np.unique(edges, axis=0, return_counts=True)
    unreliable_vertices = np.unique(unique_edges[edge_counts != 2])
    reliable_faces = valid_faces.copy()
    reliable_faces[valid_faces] &= ~np.isin(
        valid_face_indices, unreliable_vertices
    ).any(axis=1)
    reliable_area = float(face_areas[reliable_faces].sum())
    total_area = float(face_areas[valid_faces].sum())
    if reliable_area <= 0 or total_area <= 0:
        raise ValueError("Mesh has no reliable interior manifold area")

    reliable_triangles = triangles[reliable_faces]
    reliable_face_areas = face_areas[reliable_faces]
    rng = np.random.default_rng(seed)
    support_count = max(
        CURVATURE_MIN_SUPPORT_SAMPLES,
        int(np.ceil(CURVATURE_SUPPORT_SAMPLES_PER_AREA * reliable_area)),
    )
    support_points = _sample_surface(
        reliable_triangles, reliable_face_areas, support_count, rng
    )
    query_points = _sample_surface(
        reliable_triangles, reliable_face_areas, query_sample_count, rng
    )
    distances, neighbors = cKDTree(support_points).query(
        query_points,
        k=CURVATURE_MAX_NEIGHBORS,
        distance_upper_bound=radius,
        workers=-1,
    )
    curvature_values = []
    for point, distance, neighbor_index in zip(query_points, distances, neighbors):
        valid_neighbors = np.isfinite(distance)
        if np.count_nonzero(valid_neighbors) < CURVATURE_MIN_NEIGHBORS:
            continue
        local_points = support_points[neighbor_index[valid_neighbors]]
        curvature = _fit_mean_curvature(point, local_points, radius)
        if curvature is not None:
            curvature_values.append(curvature)

    fit_fraction = len(curvature_values) / query_sample_count
    if not curvature_values:
        raise ValueError("No local curvature fits succeeded")
    curvature_values = np.minimum(curvature_values, CURVATURE_MAX)
    histogram_values = np.log1p(curvature_values)
    histogram, _ = np.histogram(histogram_values, bins=CURVATURE_BIN_EDGES)
    histogram = histogram.astype(np.float64)
    histogram /= histogram.sum()
    return histogram, min(reliable_area / total_area, 1.0), fit_fraction


# Database-wide curvature extraction and quality reporting

def extract_curvature_descriptors(database_path, seed=42,
                                  min_fit_fraction=CURVATURE_MIN_FIT_FRACTION):
    """Extract curvature histograms and quality fractions for a processed DB.

    Args:
        database_path: Root of the preprocessed database.
        seed: Base seed combined with each stable relative mesh identifier.
        min_fit_fraction: Threshold used to label low-fit rows; histogram and
            measured fraction are retained whenever any local fits succeed.
    Returns:
        DataFrame with class/file keys, bin proportions, coverage, status, errors.
    """
    database_path = Path(database_path).resolve()
    rows = []
    for mesh_path in tqdm(
        mmru.find_obj_files(database_path), desc="Curvature descriptors"
    ):
        relative = mesh_path.relative_to(database_path)
        row = {
            "Class": relative.parent.name,
            "File": relative.name,
            **{column: np.nan for column in CURVATURE_COLUMNS},
            "Interior_manifold_area_fraction": np.nan,
            "Successful_fit_sample_fraction": np.nan,
            "Status": "ERROR",
            "Error": "",
        }
        try:
            identifier = mmru.mesh_identifier(mesh_path, database_path)
            histogram, coverage, fit_fraction = curvature_single(
                mmru.load_mesh(mesh_path),
                seed=mmru.stable_seed(identifier, seed),
            )
            row.update(dict(zip(CURVATURE_COLUMNS, histogram)))
            row["Interior_manifold_area_fraction"] = coverage
            row["Successful_fit_sample_fraction"] = fit_fraction
            row["Status"] = "OK" if fit_fraction >= min_fit_fraction else "LOW_FIT"
            if row["Status"] == "LOW_FIT":
                row["Error"] = (
                    f"Fit fraction {fit_fraction:.2%} is below "
                    f"{min_fit_fraction:.2%}"
                )
        except Exception as error:
            row["Error"] = str(error)
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    """Extract curvature distributions and save coverage/status diagnostics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_DIR / "shapes" / "ShapeDatabase_INFOMR_cleaned_sample",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=PROJECT_DIR / "features" / "sample",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--minimum-fit-fraction",
        type=float,
        default=CURVATURE_MIN_FIT_FRACTION,
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = args.features / "curvature_descriptors.csv"
    if output.exists() and not args.overwrite:
        parser.error(
            "Curvature output already exists; pass --overwrite to replace it."
        )
    if not 0.0 <= args.minimum_fit_fraction <= 1.0:
        parser.error("--minimum-fit-fraction must be between 0 and 1")
    args.features.mkdir(parents=True, exist_ok=True)
    descriptors = extract_curvature_descriptors(
        args.database, seed=args.seed,
        min_fit_fraction=args.minimum_fit_fraction,
    )
    descriptors.to_csv(output, index=False)
    config_path = mmru.write_feature_config(args.features, "curvature", {
        "seed": args.seed,
        "min_fit_fraction": args.minimum_fit_fraction,
        "radius": CURVATURE_RADIUS,
        "query_sample_count": CURVATURE_QUERY_SAMPLE_COUNT,
        "maximum_curvature": CURVATURE_MAX,
        "bin_count": CURVATURE_BIN_COUNT,
    })
    print(f"Meshes: {len(descriptors)}")
    print(f"Successful fits: {(descriptors['Status'] == 'OK').sum()}")
    print(f"Below fit threshold: {(descriptors['Status'] == 'LOW_FIT').sum()}")
    print(f"Errors: {(descriptors['Status'] == 'ERROR').sum()}")
    print(f"Curvature histograms and coverage: {output}")
    print(f"Feature settings: {config_path}")


if __name__ == "__main__":
    main()
