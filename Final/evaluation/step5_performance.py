"""Benchmark exact brute-force and cKDTree KNN over saved mesh features."""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


FINAL_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = FINAL_DIR.parent
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))

from curvature_descriptors import CURVATURE_BIN_COUNT
from elementary_descriptors import ELEMENTARY_COLUMNS
from histogram_descriptors import (
    HISTOGRAM_BIN_COUNT,
    HISTOGRAM_DESCRIPTORS,
)
from query import DESCRIPTOR_WEIGHTS
from scalability import FeatureSearchIndex


def _query_features_from_candidate(index, row):
    """Use one saved feature row as a deterministic database query."""
    scaled = {}
    for feature in ELEMENTARY_COLUMNS:
        scaler_row = index.scaler.loc[feature]
        scaled[feature] = (
            float(row[feature]) - float(scaler_row["Mean"])
        ) / float(scaler_row["Std"])

    histograms = {}
    for descriptor in HISTOGRAM_DESCRIPTORS + ["Curvature"]:
        bin_count = (
            CURVATURE_BIN_COUNT
            if descriptor == "Curvature"
            else HISTOGRAM_BIN_COUNT
        )
        histograms[descriptor] = np.array([
            row[f"{descriptor}_bin_{bin_index}"]
            for bin_index in range(bin_count)
        ], dtype=np.float64)
    return scaled, histograms


def _identity(result):
    return result["Class"], result["File"]


def run_benchmark(
    database_path,
    features_path,
    output_dir,
    query_count=25,
    max_k=30,
    repeats=1,
    seed=42,
):
    """Benchmark search time and verify KNN results against brute force."""
    start = time.perf_counter()
    index = FeatureSearchIndex(
        database_path, features_path, DESCRIPTOR_WEIGHTS
    )
    index_build_seconds = time.perf_counter() - start
    if index.tree is None:
        raise ValueError(
            "Cannot benchmark cKDTree because the feature index is "
            f"incomplete: {index.tree_unavailable_reason}"
        )

    candidate_count = len(index.candidates)
    if candidate_count < 2:
        raise ValueError("At least two usable candidates are required")
    if query_count < 1:
        raise ValueError("Query count must be at least one")
    if max_k < 1:
        raise ValueError("Maximum K must be at least one")
    if repeats < 1:
        raise ValueError("Repeat count must be at least one")

    query_count = min(query_count, candidate_count)
    max_k = min(max_k, candidate_count - 1)
    rng = np.random.default_rng(seed)
    query_indices = rng.choice(
        candidate_count, size=query_count, replace=False
    )
    k_values = np.arange(1, max_k + 1, dtype=int)
    brute_times = []
    knn_times = {int(k): [] for k in k_values}
    exact_matches = {int(k): 0 for k in k_values}
    comparison_count = 0

    for query_index in query_indices:
        row = index.candidates.iloc[int(query_index)]
        query_scaled, query_histograms = _query_features_from_candidate(
            index, row
        )
        query_key = (row["Class"], row["File"])
        brute_results = None
        repeated_brute_times = []
        for _ in range(repeats):
            start = time.perf_counter()
            brute_results, _ = index.search(
                query_scaled,
                query_histograms,
                search_method="brute_force",
                top_k=max_k,
                query_key=query_key,
            )
            repeated_brute_times.append(time.perf_counter() - start)
        brute_times.append(float(np.median(repeated_brute_times)))

        for k in k_values:
            repeated_knn_times = []
            knn_results = None
            for _ in range(repeats):
                start = time.perf_counter()
                knn_results, info = index.search(
                    query_scaled,
                    query_histograms,
                    search_method="knn",
                    top_k=int(k),
                    query_key=query_key,
                )
                repeated_knn_times.append(time.perf_counter() - start)
            if info["Search_method_used"] != "knn":
                raise RuntimeError(
                    "KNN benchmark unexpectedly fell back to brute force: "
                    f"{info['Search_fallback_reason']}"
                )
            knn_times[int(k)].append(
                float(np.median(repeated_knn_times))
            )
            expected = [_identity(result) for result in brute_results[:k]]
            actual = [_identity(result) for result in knn_results]
            exact_matches[int(k)] += int(actual == expected)
            comparison_count += 1

    mean_brute_ms = float(np.mean(brute_times) * 1000)
    rows = []
    for k in k_values:
        mean_knn_ms = float(np.mean(knn_times[int(k)]) * 1000)
        exact_match_rate = exact_matches[int(k)] / query_count
        rows.append({
            "K": int(k),
            "Query_count": query_count,
            "Brute_force_ms_per_query": mean_brute_ms,
            "cKDTree_ms_per_query": mean_knn_ms,
            "Speedup": mean_brute_ms / mean_knn_ms,
            "Exact_match_rate": exact_match_rate,
            "Index_build_seconds": index_build_seconds,
        })

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "step5_performance_results.csv"
    plot_path = output_dir / "step5_performance_plot.png"
    results = pd.DataFrame(rows)
    results.to_csv(results_path, index=False)

    figure, axis = plt.subplots(figsize=(9, 6))
    axis.plot(
        results["K"],
        results["Brute_force_ms_per_query"],
        label="Brute force (query.py score)",
    )
    axis.plot(
        results["K"],
        results["cKDTree_ms_per_query"],
        label="cKDTree KNN (exact, weighted L1)",
    )
    axis.set_yscale("log")
    axis.set_xlabel("Number of nearest neighbors (K)")
    axis.set_ylabel("Mean search time per query (ms)")
    axis.set_title(
        f"Search performance over {query_count} sampled queries"
    )
    axis.grid(True, which="both", alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_path, dpi=200)
    plt.close(figure)

    overall_match_rate = (
        sum(exact_matches.values()) / comparison_count
        if comparison_count else 1.0
    )
    print(f"Usable candidates: {candidate_count}")
    print(f"Sampled queries: {query_count}")
    print(f"Feature loading and index build: {index_build_seconds:.3f} s")
    print(f"Mean brute-force search: {mean_brute_ms:.3f} ms/query")
    print(
        "Mean exact-match rate across K values: "
        f"{overall_match_rate:.1%}"
    )
    print(f"Saved timing data: {results_path}")
    print(f"Saved performance plot: {plot_path}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=FINAL_DIR / "shapes" / "ShapeDatabase_INFOMR_cleaned",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=FINAL_DIR / "features",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=FINAL_DIR / "evaluation" / "results",
    )
    parser.add_argument(
        "--queries",
        type=int,
        default=25,
        help="Number of database meshes to sample as queries.",
    )
    parser.add_argument("--max-k", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_benchmark(
        args.database,
        args.features,
        args.output_dir,
        query_count=args.queries,
        max_k=args.max_k,
        repeats=args.repeats,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
