"""Extract normalized A3 and D1-D4 histograms with shared database bins."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import MMR_utility as mmru


HISTOGRAM_SAMPLE_COUNT = 10000
HISTOGRAM_BIN_COUNT = 25
HISTOGRAM_DESCRIPTORS = ["D1", "D2", "D3", "D4", "A3"]
PROJECT_DIR = Path(__file__).resolve().parent


# Per-mesh sampling and histogram construction

def _distinct_vertex_indices(rng, sample_count, vertices_count, arity):
    """Draw rows of vertex indices without repeats within each sample."""
    if vertices_count < arity:
        raise ValueError(f"Descriptor requires at least {arity} vertices")
    indices = rng.integers(0, vertices_count, size=(sample_count, arity))
    while True:
        repeated = np.zeros(sample_count, dtype=bool)
        for left in range(arity):
            for right in range(left + 1, arity):
                repeated |= indices[:, left] == indices[:, right]
        if not np.any(repeated):
            return indices
        indices[repeated] = rng.integers(
            0, vertices_count, size=(int(repeated.sum()), arity)
        )


def sample_histogram_values(mesh, sample_count=HISTOGRAM_SAMPLE_COUNT, seed=42):
    """Sample raw A3 and D1-D4 values from preprocessed mesh vertices.

    Args:
        mesh: Preprocessed normalized Trimesh.
        sample_count: Samples drawn for each descriptor.
        seed: Stable per-mesh seed shared between database and query.
    Returns:
        Mapping from descriptor name to one-dimensional sample arrays.
    """
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    rng = np.random.default_rng(seed)
    vertex_count = len(vertices)
    d1_points = vertices[rng.integers(0, vertex_count, size=sample_count)]
    d1 = np.linalg.norm(d1_points, axis=1)

    d2_indices = _distinct_vertex_indices(rng, sample_count, vertex_count, 2)
    d2 = np.linalg.norm(
        vertices[d2_indices[:, 0]] - vertices[d2_indices[:, 1]], axis=1
    )

    d3_indices = _distinct_vertex_indices(rng, sample_count, vertex_count, 3)
    d3_first = vertices[d3_indices[:, 1]] - vertices[d3_indices[:, 0]]
    d3_second = vertices[d3_indices[:, 2]] - vertices[d3_indices[:, 0]]
    d3_area = np.linalg.norm(np.cross(d3_first, d3_second), axis=1) / 2.0
    d3 = np.sqrt(d3_area)

    d4_indices = _distinct_vertex_indices(rng, sample_count, vertex_count, 4)
    d4_first = vertices[d4_indices[:, 1]] - vertices[d4_indices[:, 0]]
    d4_second = vertices[d4_indices[:, 2]] - vertices[d4_indices[:, 0]]
    d4_third = vertices[d4_indices[:, 3]] - vertices[d4_indices[:, 0]]
    d4_volume = np.abs(
        np.einsum("ij,ij->i", d4_first, np.cross(d4_second, d4_third))
    ) / 6.0
    d4 = np.cbrt(d4_volume)

    a3_indices = _distinct_vertex_indices(rng, sample_count, vertex_count, 3)
    first = vertices[a3_indices[:, 0]]
    second = vertices[a3_indices[:, 1]]
    third = vertices[a3_indices[:, 2]]
    first_vector = second - first
    second_vector = third - first
    denominator = (
        np.linalg.norm(first_vector, axis=1)
        * np.linalg.norm(second_vector, axis=1)
    )
    while np.any(denominator <= 0):
        invalid = denominator <= 0
        a3_indices[invalid] = _distinct_vertex_indices(
            rng, int(invalid.sum()), vertex_count, 3
        )
        first = vertices[a3_indices[:, 0]]
        second = vertices[a3_indices[:, 1]]
        third = vertices[a3_indices[:, 2]]
        first_vector = second - first
        second_vector = third - first
        denominator = (
            np.linalg.norm(first_vector, axis=1)
            * np.linalg.norm(second_vector, axis=1)
        )
    cosine = np.einsum("ij,ij->i", first_vector, second_vector) / denominator
    a3 = np.arccos(np.clip(cosine, -1.0, 1.0))
    return {"D1": d1, "D2": d2, "D3": d3, "D4": d4, "A3": a3}


def histogram_edges_from_samples(sample_ranges, bin_count=HISTOGRAM_BIN_COUNT):
    """Create shared linear bin edges from database-wide sample extrema.

    Args:
        sample_ranges: Mapping descriptor name to observed min/max pair.
        bin_count: Number of bins for each descriptor.
    Returns:
        Mapping from descriptor to a fixed edge array; A3 uses [0, pi].
    """
    edges = {"A3": np.linspace(0.0, np.pi, bin_count + 1)}
    for descriptor in ("D1", "D2", "D3", "D4"):
        minimum, maximum = sample_ranges[descriptor]
        if not np.isfinite(minimum) or not np.isfinite(maximum):
            raise ValueError(f"No finite samples for histogram {descriptor}")
        if maximum <= minimum:
            maximum = minimum + max(abs(minimum) * 1e-9, 1e-9)
        edges[descriptor] = np.linspace(minimum, maximum, bin_count + 1)
    return edges


def histogram_from_samples(samples, edges):
    """Convert samples to a normalized histogram using fixed shared edges.

    Args:
        samples: Raw descriptor samples.
        edges: Fixed database-derived bin edges.
    Returns:
        One-dimensional bin proportions that sum to one.
    """
    clipped = np.clip(np.asarray(samples, dtype=np.float64), edges[0], edges[-1])
    counts, _ = np.histogram(clipped, bins=edges)
    if counts.sum() <= 0:
        raise ValueError("Histogram has no samples inside its bin range")
    return counts.astype(np.float64) / counts.sum()


def histogram_edges_frame(edges):
    """Serialize histogram bin edges and centers to a tidy DataFrame.

    Args:
        edges: Mapping returned by histogram_edges_from_samples.
    Returns:
        DataFrame with descriptor, bin index, left/right edges, and center.
    """
    rows = []
    for descriptor in HISTOGRAM_DESCRIPTORS:
        descriptor_edges = edges[descriptor]
        for index in range(len(descriptor_edges) - 1):
            left = float(descriptor_edges[index])
            right = float(descriptor_edges[index + 1])
            rows.append({
                "Descriptor": descriptor,
                "Bin": index,
                "Left": left,
                "Right": right,
                "Center": (left + right) / 2.0,
            })
    return pd.DataFrame(rows)


def load_histogram_edges(edges_path):
    """Load fixed histogram edges saved by this module's extraction script.

    Args:
        edges_path: CSV containing Descriptor, Bin, Left, and Right columns.
    Returns:
        Mapping from each histogram descriptor to an ordered edge array.
    """
    frame = pd.read_csv(edges_path)
    edges = {}
    for descriptor in HISTOGRAM_DESCRIPTORS:
        rows = frame.loc[frame["Descriptor"] == descriptor].sort_values("Bin")
        if len(rows) != HISTOGRAM_BIN_COUNT:
            raise ValueError(f"Expected {HISTOGRAM_BIN_COUNT} bins for {descriptor}")
        edges[descriptor] = np.concatenate((
            rows["Left"].to_numpy(dtype=float)[:1],
            rows["Right"].to_numpy(dtype=float),
        ))
    return edges


# Database-wide shared bin fitting and feature extraction

def extract_histogram_descriptors(database_path,
                                  sample_count=HISTOGRAM_SAMPLE_COUNT,
                                  seed=42, bin_count=HISTOGRAM_BIN_COUNT):
    """Extract normalized A3/D1-D4 histograms in two deterministic passes.

    Args:
        database_path: Root of the preprocessed database.
        sample_count: Samples drawn for each descriptor and mesh.
        seed: Base seed combined with each stable relative mesh identifier.
        bin_count: Number of shared bins per descriptor.
    Returns:
        Tuple of histogram DataFrame and fixed-edge DataFrame.
    """
    database_path = Path(database_path).resolve()
    mesh_paths = mmru.find_obj_files(database_path)
    if not mesh_paths:
        raise ValueError(f"No OBJ meshes found under {database_path}")

    ranges = {name: [np.inf, -np.inf] for name in HISTOGRAM_DESCRIPTORS}
    failures = {}
    for mesh_path in tqdm(mesh_paths, desc="Histogram range pass"):
        identifier = mmru.mesh_identifier(mesh_path, database_path)
        try:
            samples = sample_histogram_values(
                mmru.load_mesh(mesh_path), sample_count,
                mmru.stable_seed(identifier, seed),
            )
            for descriptor, values in samples.items():
                ranges[descriptor][0] = min(
                    ranges[descriptor][0], float(values.min())
                )
                ranges[descriptor][1] = max(
                    ranges[descriptor][1], float(values.max())
                )
        except Exception as error:
            failures[mesh_path] = str(error)
    edges = histogram_edges_from_samples(ranges, bin_count)

    rows = []
    for mesh_path in tqdm(mesh_paths, desc="Histogram extraction"):
        relative = mesh_path.relative_to(database_path)
        row = {
            "Class": relative.parent.name,
            "File": relative.name,
            **{
                f"{descriptor}_bin_{index}": np.nan
                for descriptor in HISTOGRAM_DESCRIPTORS
                for index in range(bin_count)
            },
            "Status": "ERROR",
            "Error": failures.get(mesh_path, ""),
        }
        if mesh_path in failures:
            rows.append(row)
            continue
        try:
            samples = sample_histogram_values(
                mmru.load_mesh(mesh_path), sample_count,
                mmru.stable_seed(
                    mmru.mesh_identifier(mesh_path, database_path), seed
                ),
            )
            for descriptor in HISTOGRAM_DESCRIPTORS:
                histogram = histogram_from_samples(
                    samples[descriptor], edges[descriptor]
                )
                row.update({
                    f"{descriptor}_bin_{index}": value
                    for index, value in enumerate(histogram)
                })
            row["Status"] = "OK"
        except Exception as error:
            row["Error"] = str(error)
        rows.append(row)
    return pd.DataFrame(rows), histogram_edges_frame(edges)


def main():
    """Extract normalized shape histograms and their shared bin locations."""
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
    parser.add_argument("--sample-count", type=int, default=HISTOGRAM_SAMPLE_COUNT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outputs = [
        args.features / "histogram_descriptors.csv",
        args.features / "histogram_bins.csv",
    ]
    if not args.overwrite and any(path.exists() for path in outputs):
        parser.error(
            "Histogram outputs already exist; pass --overwrite to replace them."
        )
    args.features.mkdir(parents=True, exist_ok=True)
    descriptors, bins = extract_histogram_descriptors(
        args.database, sample_count=args.sample_count, seed=args.seed
    )
    descriptors.to_csv(outputs[0], index=False)
    bins.to_csv(outputs[1], index=False)
    config_path = mmru.write_feature_config(args.features, "histogram", {
        "sample_count": args.sample_count,
        "seed": args.seed,
        "bin_count": HISTOGRAM_BIN_COUNT,
    })
    successful = int((descriptors["Status"] == "OK").sum())
    print(f"Meshes: {len(descriptors)}; successful: {successful}")
    print(f"Normalized histogram proportions: {outputs[0]}")
    print(f"Shared bin definitions: {outputs[1]}")
    print(f"Feature settings: {config_path}")


if __name__ == "__main__":
    main()
