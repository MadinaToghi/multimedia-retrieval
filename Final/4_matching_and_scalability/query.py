"""Preprocess one OBJ, rank saved features, and browse the closest matches."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
for dependency_folder in (
    "1_read_data",
    "2_preprocess",
    "3_feature_extraction",
    "4_matching_and_scalability",
):
    sys.path.insert(0, str(PROJECT_DIR / dependency_folder))

import MMR_utility as mmru
from curvature_descriptors import (
    curvature_single,
)
from elementary_descriptors import (
    ELEMENTARY_COLUMNS,
    elementary_descriptors,
    standardize_elementary,
)
from histogram_descriptors import (
    histogram_from_samples,
    load_histogram_edges,
    sample_histogram_values,
)
from preprocess import preprocess_query_mesh
from scalability import FeatureSearchIndex


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
def query_database(query_path, database_path, features_path, source_path=None,
                   top_k=5, seed=None, search_method="brute_force",
                   radius=None, search_index=None):
    """Process a query and search saved database features.

    Args:
        query_path: Raw query OBJ path.
        database_path: Matching preprocessed database root.
        features_path: Directory containing feature and scaler CSVs.
        source_path: Original database root used for stable query identity.
        top_k: Number of nearest candidates to return.
        seed: Optional override for saved histogram and curvature seeds.
        search_method: One of brute_force, knn, or rnn.
        radius: Maximum combined feature distance for RNN search.
        search_index: Optional preloaded database feature index.
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
    if search_index is None:
        search_index = FeatureSearchIndex(
            database_path, features_path, DESCRIPTOR_WEIGHTS
        )
    elif (
        search_index.database_path != database_path
        or search_index.features_path != features_path.resolve()
    ):
        raise ValueError(
            "The supplied search index does not match the selected database "
            "and feature paths"
        )
    ranked, search_info = search_index.search(
        query_scaled,
        query_histograms,
        search_method=search_method,
        top_k=top_k,
        radius=radius,
        query_key=query_database_key,
    )
    return query_mesh, {
        "Preprocessing": preprocess_info,
        "Curvature_coverage": curvature_coverage,
        "Curvature_fit_fraction": curvature_fit_fraction,
        "Curvature_error": curvature_error,
        "Query_feature_values": query_feature_values,
        "Identifier": identifier,
        "Histogram_seed": histogram_seed,
        "Curvature_seed": curvature_seed,
        "Elementary_quality_warnings": query_elementary["Quality_Warnings"],
        **search_info,
    }, ranked


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
    displayed_matches = matches[:5]
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
    for rank, match in enumerate(displayed_matches, start=1):
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
        for match in displayed_matches:
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
            for rank, match in enumerate(displayed_matches, start=1)
        ],
        cellLoc="center",
        loc="center",
        colWidths=[0.13, 0.12] + [0.15] * len(displayed_matches),
    )
    plot_title = (
        f"Query matches: {query_path.name} | "
        f"curvature coverage {query_quality['Curvature_coverage']:.1%}, "
        f"fit success {query_quality['Curvature_fit_fraction']:.1%}"
    )
    if len(matches) > len(displayed_matches):
        plot_title += f" | showing first {len(displayed_matches)} of {len(matches)}"
    figure.suptitle(
        plot_title
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
        "--search-method",
        choices=("brute_force", "knn", "rnn"),
        default=None,
        help="Search strategy; RNN requires --radius.",
    )
    parser.add_argument(
        "--radius",
        type=float,
        help="Maximum combined feature score for RNN search.",
    )
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
    search_method = args.search_method or "brute_force"
    if search_method != "rnn" and args.top_k < 1:
        parser.error("--top-k must be at least one")
    if search_method == "rnn":
        if args.radius is None or not np.isfinite(args.radius) or args.radius < 0:
            parser.error("--radius must be a finite non-negative number for RNN")
    elif args.radius is not None:
        parser.error("--radius can only be used with --search-method rnn")

    if not (
        args.cli
        or args.mesh is not None
        or args.no_plot
        or args.search_method is not None
        or args.radius is not None
    ):
        from query_ui import launch_query_ui

        search_index = FeatureSearchIndex(
            args.database, args.features, DESCRIPTOR_WEIGHTS
        )
        launch_query_ui(
            query_database=query_database,
            descriptor_weights=DESCRIPTOR_WEIGHTS,
            database_path=args.database,
            source_path=args.source,
            features_path=args.features,
            top_k=args.top_k,
            seed=args.seed,
            search_index=search_index,
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
        search_method=search_method,
        radius=args.radius,
    )
    print(f"Query: {mesh_path}")
    print(f"Candidate meshes ranked: {quality['Candidates_ranked']}")
    print(f"Search method: {quality['Search_method_used']}")
    if quality["Search_fallback_reason"]:
        print(
            "Exact brute-force fallback: "
            f"{quality['Search_fallback_reason']}"
        )
    if search_method == "rnn":
        print(
            f"Matches within radius {args.radius:g}: {len(matches)}"
        )
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
