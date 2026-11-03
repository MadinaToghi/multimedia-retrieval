"""Compute and plot area-sampled mean-curvature histograms for the dataset."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import trimesh as tr
from matplotlib.backends.backend_pdf import PdfPages
from scipy.spatial import cKDTree
from tqdm import tqdm


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = (PROJECT_DIR / "../ShapeDatabase_INFOMR_normalized_cleaned").resolve()
CSV_PATH = PROJECT_DIR / "curvature_histograms.csv"
PLOT_PATH = PROJECT_DIR / "curvature_histograms_by_category.pdf"

BIN_COUNT = 25
MAX_CURVATURE = 100.0
MIN_FACE_AREA = 1e-14
NEIGHBORHOOD_RADIUS = 0.05
QUERY_SAMPLE_COUNT = 3000
SUPPORT_SAMPLES_PER_AREA = 6000
MIN_SUPPORT_SAMPLE_COUNT = 12000
MIN_NEIGHBOR_COUNT = 12
MAX_NEIGHBOR_COUNT = 50
MIN_SUCCESSFUL_FIT_FRACTION = 0.95
BIN_EDGES = np.linspace(0.0, np.log1p(MAX_CURVATURE), BIN_COUNT + 1)
BIN_CENTERS = (BIN_EDGES[:-1] + BIN_EDGES[1:]) / 2.0
HISTOGRAM_COLUMNS = [
    f"Curvature_bin_{index}" for index in range(BIN_COUNT)
]


def _sample_surface(triangles, areas, count, rng):
    """Draw points uniformly by area from a set of triangles."""
    face_indices = rng.choice(
        len(triangles), size=count, p=areas / areas.sum()
    )
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
    """Fit a local quadratic patch and return absolute mean curvature."""
    centered = neighbors - point
    covariance = centered.T @ centered / len(centered)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[1] <= np.finfo(np.float64).eps * radius**2:
        return None

    # The two dominant principal directions span the local tangent plane.
    local_coordinates = centered @ eigenvectors[:, [2, 1, 0]]
    x, y, z = (local_coordinates[:, axis] / radius for axis in range(3))
    radial_distance_squared = x * x + y * y
    weights = np.sqrt(np.exp(-2.0 * radial_distance_squared))
    design = np.column_stack(
        (np.ones(len(x)), x, y, 0.5 * x * x, x * y, 0.5 * y * y)
    )
    coefficients, _, rank, _ = np.linalg.lstsq(
        design * weights[:, None],
        z * weights,
        rcond=None,
    )
    if rank < 6:
        return None

    slope_x, slope_y = coefficients[1:3] / radius
    second_xx, second_xy, second_yy = coefficients[3:6] / radius
    mean_curvature = (
        (1.0 + slope_y**2) * second_xx
        - 2.0 * slope_x * slope_y * second_xy
        + (1.0 + slope_x**2) * second_yy
    ) / (
        2.0
        * (1.0 + slope_x**2 + slope_y**2) ** 1.5
    )
    if not np.isfinite(mean_curvature):
        return None
    return abs(mean_curvature)


def curvature_single(mesh, seed=0, radius=NEIGHBORHOOD_RADIUS):
    """Return a histogram and topological/fit coverage for one normalized mesh.

    Curvature is estimated from quadratic patches fit to area-uniform surface
    samples within a fixed physical radius. Vertices touching boundary or
    non-manifold edges are excluded from the sample domain.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(vertices) == 0 or len(faces) == 0:
        raise ValueError("Curvature requires a non-empty triangular mesh")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError("Curvature requires triangular faces")
    if not np.all(np.isfinite(vertices)):
        raise ValueError("Mesh vertices must be finite")
    if not np.isfinite(radius) or radius <= 0:
        raise ValueError("Neighborhood radius must be positive and finite")

    maximum_extent = float(np.max(np.ptp(vertices, axis=0)))
    if not np.isclose(maximum_extent, 1.0, rtol=0.0, atol=1e-3):
        raise ValueError(
            "Curvature histogram requires a mesh normalized to unit extent"
        )

    triangles = vertices[faces]
    face_areas = np.linalg.norm(
        np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        ),
        axis=1,
    ) / 2.0
    valid_faces = np.isfinite(face_areas) & (face_areas > MIN_FACE_AREA)
    if not np.any(valid_faces):
        raise ValueError("Mesh has no non-degenerate triangles")

    # Edges not shared by exactly two valid triangles are topologically unreliable.
    valid_face_indices = faces[valid_faces]
    edges = np.concatenate(
        (
            valid_face_indices[:, [0, 1]],
            valid_face_indices[:, [1, 2]],
            valid_face_indices[:, [2, 0]],
        ),
        axis=0,
    )
    edges.sort(axis=1)
    unique_edges, edge_counts = np.unique(
        edges, axis=0, return_counts=True
    )
    unreliable_vertices = np.unique(unique_edges[edge_counts != 2])
    reliable_faces = valid_faces.copy()
    reliable_faces[valid_faces] &= ~np.isin(
        valid_face_indices, unreliable_vertices
    ).any(axis=1)

    reliable_area = float(face_areas[reliable_faces].sum())
    total_area = float(face_areas[valid_faces].sum())
    if reliable_area <= 0 or total_area <= 0:
        raise ValueError("Mesh has no interior manifold surface area")

    reliable_triangles = triangles[reliable_faces]
    reliable_face_areas = face_areas[reliable_faces]
    rng = np.random.default_rng(seed)
    support_count = max(
        MIN_SUPPORT_SAMPLE_COUNT,
        int(np.ceil(SUPPORT_SAMPLES_PER_AREA * reliable_area)),
    )
    support_points = _sample_surface(
        reliable_triangles,
        reliable_face_areas,
        support_count,
        rng,
    )
    query_points = _sample_surface(
        reliable_triangles,
        reliable_face_areas,
        QUERY_SAMPLE_COUNT,
        rng,
    )
    distances, neighbors = cKDTree(support_points).query(
        query_points,
        k=MAX_NEIGHBOR_COUNT,
        distance_upper_bound=radius,
        workers=-1,
    )

    curvature_values = []
    for point, distance, neighbor_index in zip(
        query_points, distances, neighbors
    ):
        valid_neighbors = np.isfinite(distance)
        if np.count_nonzero(valid_neighbors) < MIN_NEIGHBOR_COUNT:
            continue
        local_points = support_points[neighbor_index[valid_neighbors]]
        curvature = _fit_mean_curvature(point, local_points, radius)
        if curvature is not None:
            curvature_values.append(curvature)

    fit_fraction = len(curvature_values) / QUERY_SAMPLE_COUNT
    if fit_fraction < MIN_SUCCESSFUL_FIT_FRACTION:
        raise ValueError(
            f"Only {fit_fraction:.1%} of local curvature fits succeeded; "
            f"minimum required is {MIN_SUCCESSFUL_FIT_FRACTION:.0%}"
        )

    # Meshes from read_data.py have unit maximum extent, so no scale factor is needed.
    curvature_values = np.minimum(curvature_values, MAX_CURVATURE)
    histogram_values = np.log1p(curvature_values)
    histogram, _ = np.histogram(histogram_values, bins=BIN_EDGES)
    histogram = histogram.astype(np.float64)
    histogram_total = float(histogram.sum())
    if histogram_total <= 0 or not np.isfinite(histogram_total):
        raise ValueError("Could not normalize the curvature histogram")
    histogram /= histogram_total

    return (
        histogram,
        min(reliable_area / total_area, 1.0),
        fit_fraction,
    )


def process_all_meshes():
    """Calculate histograms for every normalized OBJ and write the CSV."""
    mesh_paths = sorted(DATA_DIR.rglob("*.obj"))
    if not mesh_paths:
        raise FileNotFoundError(f"No OBJ meshes found under {DATA_DIR}")

    rows = []
    for mesh_path in tqdm(mesh_paths, desc="Curvature histograms"):
        row = {
            "Class": mesh_path.parent.name,
            "File": mesh_path.name,
            "Interior_manifold_area_fraction": np.nan,
            "Successful_fit_sample_fraction": np.nan,
            **{column: np.nan for column in HISTOGRAM_COLUMNS},
            "status": "ERROR",
            "error": "",
        }
        try:
            mesh = tr.load(mesh_path, force="mesh", process=False)
            histogram, area_fraction, fit_fraction = curvature_single(mesh)
            row["Interior_manifold_area_fraction"] = area_fraction
            row["Successful_fit_sample_fraction"] = fit_fraction
            row.update(
                {
                    column: value
                    for column, value in zip(HISTOGRAM_COLUMNS, histogram)
                }
            )
            row["status"] = "OK"
        except Exception as error:
            row["error"] = str(error)
        rows.append(row)

    columns = [
        "Class",
        "File",
        *HISTOGRAM_COLUMNS,
        "Interior_manifold_area_fraction",
        "Successful_fit_sample_fraction",
        "status",
        "error",
    ]
    results = pd.DataFrame(rows, columns=columns)
    results.to_csv(CSV_PATH, index=False, quoting=csv.QUOTE_MINIMAL)
    return results


def plot_category_histograms(results):
    """Write one PDF page per category, overlaying meshes and their mean."""
    successful = results[results["status"] == "OK"]
    if successful.empty:
        raise ValueError("No successful curvature histograms to plot")

    with PdfPages(PLOT_PATH) as pdf:
        for category, category_rows in successful.groupby("Class", sort=True):
            histograms = category_rows[HISTOGRAM_COLUMNS].to_numpy(dtype=float)
            figure, axis = plt.subplots(figsize=(9, 5))

            for histogram in histograms:
                axis.plot(
                    BIN_CENTERS,
                    histogram,
                    color="steelblue",
                    alpha=0.45,
                    linewidth=1,
                )
            axis.plot(
                BIN_CENTERS,
                histograms.mean(axis=0),
                color="darkorange",
                linewidth=2.5,
                label="Category mean",
            )
            axis.set(
                title=f"{category} curvature histograms "
                f"({len(category_rows)} meshes)",
                xlabel="log(1 + absolute mean curvature)",
                ylabel="Area-uniform normalized frequency",
                xlim=(BIN_EDGES[0], BIN_EDGES[-1]),
                ylim=(0, None),
            )
            axis.grid(axis="y", alpha=0.25)
            axis.legend()
            figure.tight_layout()
            pdf.savefig(figure)
            plt.close(figure)


def main():
    if not DATA_DIR.is_dir():
        raise FileNotFoundError(f"Normalized mesh directory not found: {DATA_DIR}")

    results = process_all_meshes()
    plot_category_histograms(results)

    errors = results[results["status"] != "OK"]
    print(f"Processed {len(results)} meshes.")
    print(f"Histograms: {CSV_PATH}")
    print(f"Category plots: {PLOT_PATH}")
    if not errors.empty:
        print(f"Failed meshes: {len(errors)}")
        for _, row in errors.iterrows():
            print(f"  {row['Class']}/{row['File']}: {row['error']}")
        raise RuntimeError(
            f"{len(errors)} mesh(es) did not produce curvature histograms"
        )
    print("All meshes produced normalized 25-bin histograms.")


if __name__ == "__main__":
    main()
