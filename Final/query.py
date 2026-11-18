"""Preprocess one OBJ, rank saved features, and browse the closest matches."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

import MMR_utility as mmru
from curvature_descriptors import (
    CURVATURE_BIN_CENTERS,
    CURVATURE_BIN_COUNT,
    curvature_single,
)
from elementary_descriptors import (
    ELEMENTARY_COLUMNS,
    ELEMENTARY_VALIDITY_COLUMNS,
    elementary_descriptors,
    standardize_elementary,
)
from histogram_descriptors import (
    HISTOGRAM_BIN_COUNT,
    HISTOGRAM_DESCRIPTORS,
    histogram_from_samples,
    load_histogram_edges,
    sample_histogram_values,
)
from preprocess import preprocess_query_mesh


# Weights are query-ranking policy, not feature extraction settings.
DESCRIPTOR_WEIGHTS = {
    "Volume": 0.7,
    "Surface Area": 1.0,
    "Compactness": 1.0,
    "Rectangularity": 1.0,
    "Diameter": 0.8,
    "Convexity": 0.6,
    "Eccentricity": 0.7,
    "D1": 1.0,
    "D2": 1.0,
    "D3": 1.3,
    "D4": 0.5,
    "A3": 0.6,
    "Curvature": 1.0,
}
PROJECT_DIR = Path(__file__).resolve().parent


def distribution_distance(query_histogram, candidate_histogram, centers):
    """Compute 1-D Wasserstein distance between normalized bin proportions.

    Args:
        query_histogram: Query bin proportions.
        candidate_histogram: Candidate bin proportions.
        centers: Shared bin-center locations.
    Returns:
        Wasserstein distance in the units of the descriptor's bin centers.
    """
    return float(wasserstein_distance(
        centers,
        centers,
        u_weights=query_histogram,
        v_weights=candidate_histogram,
    ))


def _row_histogram(row, descriptor, bin_count=HISTOGRAM_BIN_COUNT):
    """Read one row's ordered histogram bins from a pandas Series."""
    return np.array([
        row[f"{descriptor}_bin_{index}"] for index in range(bin_count)
    ], dtype=np.float64)


def query_database(query_path, database_path, features_path, source_path=None,
                   top_k=5, seed=None):
    """Process a query and rank saved database features without recomputation.

    Args:
        query_path: Raw query OBJ path.
        database_path: Matching preprocessed database root.
        features_path: Directory containing feature and scaler CSVs.
        source_path: Original database root used for stable query identity.
        top_k: Number of nearest candidates to return.
        seed: Optional override for saved histogram and curvature seeds.
    Returns:
        Tuple of canonical query mesh, query quality metadata, and ranked rows.
    """
    query_path = Path(query_path)
    database_path = Path(database_path).resolve()
    features_path = Path(features_path)
    query_mesh, preprocess_info, identifier = preprocess_query_mesh(
        query_path, seed_root=source_path
    )
    feature_config = mmru.load_feature_config(features_path)
    histogram_config = feature_config["histogram"]
    curvature_config = feature_config["curvature"]
    histogram_base_seed = histogram_config["seed"] if seed is None else seed
    curvature_base_seed = curvature_config["seed"] if seed is None else seed
    histogram_seed = mmru.stable_seed(identifier, histogram_base_seed)
    curvature_seed = mmru.stable_seed(identifier, curvature_base_seed)

    query_elementary = elementary_descriptors(query_mesh)
    scaler = pd.read_csv(features_path / "elementary_scaler.csv")
    query_scaled = standardize_elementary(query_elementary, scaler)

    bin_edges = load_histogram_edges(features_path / "histogram_bins.csv")
    sampled_values = sample_histogram_values(
        query_mesh,
        sample_count=histogram_config["sample_count"],
        seed=histogram_seed,
    )
    query_histograms = {
        descriptor: histogram_from_samples(samples, bin_edges[descriptor])
        for descriptor, samples in sampled_values.items()
    }

    curvature_error = ""
    try:
        query_curvature, curvature_coverage, curvature_fit_fraction = curvature_single(
            query_mesh,
            seed=curvature_seed,
            radius=curvature_config["radius"],
            query_sample_count=curvature_config["query_sample_count"],
        )
        query_histograms["Curvature"] = query_curvature
    except Exception as error:
        curvature_coverage = np.nan
        curvature_fit_fraction = np.nan
        curvature_error = str(error)
    query_feature_values = {
        feature: (
            float(query_elementary[feature])
            if np.isfinite(query_elementary.get(feature, np.nan))
            else None
        )
        for feature in ELEMENTARY_COLUMNS
    }
    query_feature_values.update({
        descriptor: histogram.copy()
        for descriptor, histogram in query_histograms.items()
    })

    elementary = pd.read_csv(features_path / "elementary_descriptors.csv")
    histograms = pd.read_csv(features_path / "histogram_descriptors.csv")
    curvatures = pd.read_csv(features_path / "curvature_descriptors.csv")
    keys = ["Class", "File"]
    elementary_columns = (
        ELEMENTARY_COLUMNS
        + list(ELEMENTARY_VALIDITY_COLUMNS.values())
        + ["Quality_Warnings"]
    )
    histogram_columns = [
        f"{name}_bin_{index}"
        for name in HISTOGRAM_DESCRIPTORS
        for index in range(HISTOGRAM_BIN_COUNT)
    ]
    curvature_columns = [
        f"Curvature_bin_{index}" for index in range(CURVATURE_BIN_COUNT)
    ] + [
        "Interior_manifold_area_fraction",
        "Successful_fit_sample_fraction",
    ]
    elementary = elementary[keys + elementary_columns]
    histograms = histograms[keys + histogram_columns]
    curvatures = curvatures[keys + curvature_columns]

    preprocessing_log = database_path / "preprocessing_log.csv"
    if preprocessing_log.is_file():
        candidates = pd.read_csv(preprocessing_log)[keys].drop_duplicates()
    else:
        candidates = elementary[keys].drop_duplicates()
    candidates = candidates.merge(
        elementary, on=keys, how="left", validate="one_to_one"
    ).merge(
        histograms, on=keys, how="left", validate="one_to_one"
    ).merge(
        curvatures, on=keys, how="left", validate="one_to_one"
    )
    query_database_key = None
    if source_path is not None:
        try:
            relative_query_path = query_path.resolve().relative_to(
                Path(source_path).resolve()
            )
        except ValueError:
            relative_query_path = None
        if relative_query_path is not None and len(relative_query_path.parts) > 1:
            query_database_key = (
                relative_query_path.parent.name,
                relative_query_path.name,
            )
    excluded_query_candidate = 0
    if query_database_key is not None:
        is_query_candidate = (
            candidates["Class"].eq(query_database_key[0])
            & candidates["File"].eq(query_database_key[1])
        )
        excluded_query_candidate = int(is_query_candidate.sum())
        candidates = candidates.loc[~is_query_candidate].copy()
    if candidates.empty:
        raise ValueError("No preprocessed candidates were found")

    curvature_bin_columns = [
        f"Curvature_bin_{index}" for index in range(CURVATURE_BIN_COUNT)
    ]
    has_curvature = np.isfinite(
        candidates[curvature_bin_columns].to_numpy(dtype=float)
    ).all(axis=1)
    has_curvature &= candidates[curvature_bin_columns].sum(axis=1).to_numpy() > 0
    excluded_curvature_candidates = int((~has_curvature).sum())
    candidates = candidates.loc[has_curvature].copy()
    if candidates.empty:
        raise ValueError("No candidates have a usable curvature histogram")

    scaler_index = scaler.set_index("Feature")
    bin_frame = pd.read_csv(features_path / "histogram_bins.csv")
    centers = {
        descriptor: bin_frame.loc[
            bin_frame["Descriptor"] == descriptor
        ].sort_values("Bin")["Center"].to_numpy(dtype=float)
        for descriptor in HISTOGRAM_DESCRIPTORS
    }
    centers["Curvature"] = CURVATURE_BIN_CENTERS
    descriptors = ELEMENTARY_COLUMNS + HISTOGRAM_DESCRIPTORS + ["Curvature"]
    query_weight_total = sum(
        DESCRIPTOR_WEIGHTS[name]
        for name in descriptors
        if (
            np.isfinite(query_scaled.get(name, np.nan))
            if name in ELEMENTARY_COLUMNS
            else name in query_histograms
        )
    )

    # Keep every candidate. Skip only a component whose value/histogram is absent.
    ranked = []
    for _, row in candidates.iterrows():
        distances = {name: None for name in descriptors}
        feature_values = {name: None for name in descriptors}
        unavailable = []
        for feature in ELEMENTARY_COLUMNS:
            candidate_value = row.get(feature, np.nan)
            if np.isfinite(candidate_value):
                feature_values[feature] = float(candidate_value)
            if (
                not np.isfinite(query_scaled.get(feature, np.nan))
                or not np.isfinite(candidate_value)
            ):
                unavailable.append(feature)
                continue
            candidate_value = (
                row[feature] - scaler_index.loc[feature, "Mean"]
            ) / scaler_index.loc[feature, "Std"]
            distances[feature] = abs(query_scaled[feature] - candidate_value)

        for descriptor in HISTOGRAM_DESCRIPTORS:
            candidate_histogram = _row_histogram(row, descriptor)
            if (
                not np.all(np.isfinite(candidate_histogram))
                or candidate_histogram.sum() <= 0
            ):
                unavailable.append(descriptor)
                continue
            feature_values[descriptor] = candidate_histogram.copy()
            distances[descriptor] = distribution_distance(
                query_histograms[descriptor], candidate_histogram,
                centers[descriptor],
            )

        candidate_curvature = _row_histogram(
            row, "Curvature", CURVATURE_BIN_COUNT
        )
        if (
            "Curvature" not in query_histograms
            or not np.all(np.isfinite(candidate_curvature))
            or candidate_curvature.sum() <= 0
        ):
            unavailable.append("Curvature")
        else:
            feature_values["Curvature"] = candidate_curvature.copy()
            distances["Curvature"] = distribution_distance(
                query_histograms["Curvature"], candidate_curvature,
                centers["Curvature"],
            )

        available = [
            name for name, distance in distances.items() if distance is not None
        ]
        available_weight = sum(DESCRIPTOR_WEIGHTS[name] for name in available)
        scale = query_weight_total / available_weight if available_weight else 0.0
        weighted = {
            name: (
                DESCRIPTOR_WEIGHTS[name] * distance * scale
                if distance is not None else None
            )
            for name, distance in distances.items()
        }
        score = sum(weighted[name] for name in available) if available else np.inf
        ranked.append({
            "Class": row["Class"],
            "File": row["File"],
            "Path": database_path / row["Class"] / row["File"],
            "Score": float(score),
            "Distances": distances,
            "Weighted_distances": weighted,
            "Feature_values": feature_values,
            "Unavailable_descriptors": unavailable,
            "Elementary_quality_warnings": row.get("Quality_Warnings", ""),
            "Curvature_coverage": row["Interior_manifold_area_fraction"],
            "Curvature_fit_fraction": row["Successful_fit_sample_fraction"],
        })
    ranked.sort(key=lambda result: result["Score"])
    return query_mesh, {
        "Preprocessing": preprocess_info,
        "Curvature_coverage": curvature_coverage,
        "Curvature_fit_fraction": curvature_fit_fraction,
        "Curvature_error": curvature_error,
        "Query_feature_values": query_feature_values,
        "Candidates_ranked": len(candidates),
        "Excluded_missing_curvature_candidates": excluded_curvature_candidates,
        "Excluded_query_candidate": excluded_query_candidate,
        "Identifier": identifier,
        "Histogram_seed": histogram_seed,
        "Curvature_seed": curvature_seed,
        "Elementary_quality_warnings": query_elementary["Quality_Warnings"],
    }, ranked[:top_k]


def plot_query_results(query_path, query_mesh, query_quality, matches):
    """Plot the query and top matches with per-descriptor weighted distances.

    Args:
        query_path: Original query path, used in plot labels.
        query_mesh: Canonical preprocessed query mesh.
        query_quality: Quality metadata returned by query_database.
        matches: Ranked match rows returned by query_database.
    Returns:
        The displayed Matplotlib figure.
    """
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    query_path = Path(query_path)
    figure = plt.figure(figsize=(20, 15))
    grid = figure.add_gridspec(3, 3, height_ratios=(1, 1, 1.5))
    axes = [
        figure.add_subplot(grid[row, column], projection="3d")
        for row in range(2)
        for column in range(3)
    ]
    meshes = [{
        "Label": f"Query: {query_path.parent.name}/{query_path.name}",
        "Mesh": query_mesh,
    }]
    for rank, match in enumerate(matches, start=1):
        candidate_path = Path(match["Path"])
        meshes.append({
            "Label": f"#{rank}: {match['Class']}/{match['File']} "
                     f"(score {match['Score']:.4g})",
            "Mesh": mmru.load_mesh(candidate_path) if candidate_path.is_file() else None,
        })
    for axis, item in zip(axes, meshes):
        mesh = item["Mesh"]
        if mesh is None:
            axis.set_title(f"{item['Label']}\nprocessed mesh unavailable")
            axis.set_axis_off()
            continue
        axis.add_collection3d(Poly3DCollection(
            mesh.vertices[mesh.faces],
            facecolor="lightblue",
            edgecolor="black",
            linewidth=0.1,
        ))
        center = mesh.bounds.mean(axis=0)
        radius = max(float(np.ptp(mesh.bounds, axis=0).max()) / 2.0, 1e-6)
        axis.set_xlim(center[0] - radius, center[0] + radius)
        axis.set_ylim(center[1] - radius, center[1] + radius)
        axis.set_zlim(center[2] - radius, center[2] + radius)
        axis.set_title(item["Label"])
        axis.set_axis_off()

    table_axis = figure.add_subplot(grid[2, :])
    table_axis.axis("off")
    table_rows = []
    for descriptor in DESCRIPTOR_WEIGHTS:
        values = [descriptor, f"weight: {DESCRIPTOR_WEIGHTS[descriptor]:.3g}"]
        for match in matches:
            distance = match["Distances"][descriptor]
            weighted = match["Weighted_distances"][descriptor]
            values.append(
                "n/a" if distance is None
                else f"{distance:.5g} (weighted {weighted:.5g})"
            )
        table_rows.append(values)
    table_axis.table(
        cellText=table_rows,
        colLabels=["Descriptor", "Weight"] + [
            f"#{rank} {match['Class']}/{match['File']}"
            for rank, match in enumerate(matches, start=1)
        ],
        cellLoc="center",
        loc="center",
        colWidths=[0.13, 0.12] + [0.15] * len(matches),
    )
    figure.suptitle(
        f"Query matches: {query_path.name} | "
        f"curvature coverage {query_quality['Curvature_coverage']:.1%}, "
        f"fit success {query_quality['Curvature_fit_fraction']:.1%}"
    )
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    plt.show()
    return figure


def main():
    """Open the interactive query UI, or run one query from the CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mesh",
        type=Path,
        help="Query OBJ path; supplying it runs a one-off command-line query.",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_DIR / "shapes" / "ShapeDatabase_INFOMR_cleaned",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=PROJECT_DIR / "shapes" / "ShapeDatabase_INFOMR",
        help="Original database root, used to derive the stable mesh seed.",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=PROJECT_DIR / "features",
    )
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument(
        "--seed", type=int,
        help="Optional seed override; by default saved feature settings are used.",
    )
    parser.add_argument(
        "--cli",
        action="store_true",
        help="Run a one-off command-line query instead of opening the UI.",
    )
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()
    if args.top_k < 1:
        parser.error("--top-k must be at least one")

    if not (args.cli or args.mesh is not None or args.no_plot):
        from query_ui import launch_query_ui

        launch_query_ui(
            query_database=query_database,
            descriptor_weights=DESCRIPTOR_WEIGHTS,
            database_path=args.database,
            source_path=args.source,
            features_path=args.features,
            top_k=args.top_k,
            seed=args.seed,
        )
        return

    mesh_path = args.mesh or QUERY_MESH
    query_mesh, quality, matches = query_database(
        mesh_path,
        args.database,
        args.features,
        source_path=args.source,
        top_k=args.top_k,
        seed=args.seed,
    )
    print(f"Query: {mesh_path}")
    print(f"Candidate meshes ranked: {quality['Candidates_ranked']}")
    print(f"Curvature reliable area: {quality['Curvature_coverage']:.1%}")
    print(f"Curvature fit success: {quality['Curvature_fit_fraction']:.1%}")
    if quality["Excluded_missing_curvature_candidates"]:
        print(
            "Candidates excluded for missing curvature: "
            f"{quality['Excluded_missing_curvature_candidates']}"
        )
    if quality["Excluded_query_candidate"]:
        print("Excluded the query mesh itself from candidate results")
    if quality["Elementary_quality_warnings"]:
        print(
            "Query elementary quality warnings: "
            f"{quality['Elementary_quality_warnings']}"
        )
    for rank, match in enumerate(matches, start=1):
        curvature_distance = match["Distances"]["Curvature"]
        curvature_text = (
            "n/a" if curvature_distance is None
            else f"{curvature_distance:.8g}"
        )
        print(
            f"{rank}. {match['Class']}/{match['File']} "
            f"score={match['Score']:.8g} "
            f"curvature_distance={curvature_text}"
        )
        if match["Unavailable_descriptors"]:
            print(f"   unavailable: {', '.join(match['Unavailable_descriptors'])}")
    if not args.no_plot:
        plot_query_results(mesh_path, query_mesh, quality, matches)


# The CLI runs only when this file is executed directly, not when imported.
QUERY_MESH_PATH = "ShapeDatabase_INFOMR/Car/D00793.obj"
# QUERY_MESH_PATH = "ShapeDatabase_INFOMR/Apartment/D00478.obj"
QUERY_MESH = PROJECT_DIR / "shapes" / QUERY_MESH_PATH.lstrip("/")


if __name__ == "__main__":
    main()
