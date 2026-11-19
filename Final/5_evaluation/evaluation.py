"""Evaluate current weighted retrieval with top-K classification metrics."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation_common import (
    FINAL_DIR,
    load_ranked_feature_distances,
    query_relevance,
)


def evaluate_top_k(
    database_path,
    features_path,
    output_dir,
    top_k=6,
):
    """Calculate per-query, per-class, and rank-specific retrieval metrics."""
    index, distances, ranked_indices = load_ranked_feature_distances(
        database_path, features_path
    )
    candidates = index.candidates
    classes = candidates["Class"].to_numpy(dtype=str)
    filenames = candidates["File"].to_numpy(dtype=str)
    if top_k < 1:
        raise ValueError("Top K must be at least one")

    top_k = min(top_k, len(candidates) - 1)
    query_rows = []
    rank_correct = {rank: [] for rank in range(1, top_k + 1)}
    for query_index, order in enumerate(ranked_indices):
        order = order[:top_k]
        relevant = query_relevance(classes, query_index)
        total_relevant = int(relevant.sum())
        correct = relevant[order]
        precision = float(correct.mean()) if len(correct) else np.nan
        recall = (
            float(correct.sum() / total_relevant)
            if total_relevant else np.nan
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if total_relevant and precision + recall else 0.0
        )
        query_rows.append({
            "Query_Class": classes[query_index],
            "Query_File": filenames[query_index],
            "Top_K": top_k,
            "Retrieved_Count": len(order),
            "Relevant_Count": total_relevant,
            "Correct_Count": int(correct.sum()),
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        })
        for rank, is_relevant in enumerate(correct, start=1):
            rank_correct[rank].append(bool(is_relevant))

    per_query = pd.DataFrame(query_rows)
    per_class = per_query.groupby("Query_Class", sort=True).agg(
        Query_Count=("Query_File", "count"),
        Mean_Precision=("Precision", "mean"),
        Mean_Recall=("Recall", "mean"),
        Mean_F1=("F1", "mean"),
        Median_F1=("F1", "median"),
    ).reset_index()
    summary = pd.DataFrame([{
        "Top_K": top_k,
        "Query_Count": len(per_query),
        "Candidate_Count": len(candidates),
        "Mean_Precision": per_query["Precision"].mean(),
        "Mean_Recall": per_query["Recall"].mean(),
        "Mean_F1": per_query["F1"].mean(),
        "Median_F1": per_query["F1"].median(),
    }])
    by_rank = pd.DataFrame([
        {
            "Rank": rank,
            "Query_Count": len(values),
            "Precision": float(np.mean(values)) if values else np.nan,
        }
        for rank, values in rank_correct.items()
    ])

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    per_query.to_csv(output_dir / "evaluation_results_by_query.csv", index=False)
    per_class.to_csv(output_dir / "evaluation_metrics_by_class.csv", index=False)
    summary.to_csv(output_dir / "evaluation_summary.csv", index=False)
    by_rank.to_csv(output_dir / "precision_by_rank.csv", index=False)

    figure, axis = plt.subplots(figsize=(9, 5))
    axis.bar(
        by_rank["Rank"],
        by_rank["Precision"] * 100,
        color="steelblue",
    )
    axis.set(
        xlabel="Rank",
        ylabel="Relevant results (%)",
        title=f"Per-rank precision across {len(per_query)} database queries",
        xticks=by_rank["Rank"],
        ylim=(0, 100),
    )
    axis.grid(axis="y", alpha=0.3)
    figure.tight_layout()
    figure.savefig(
        output_dir / "precision_by_rank.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Usable meshes: {len(candidates)}")
    print(f"Queries evaluated: {len(per_query)}")
    print(
        f"Top-{top_k} precision={summary['Mean_Precision'].iloc[0]:.4f}, "
        f"recall={summary['Mean_Recall'].iloc[0]:.4f}, "
        f"F1={summary['Mean_F1'].iloc[0]:.4f}"
    )
    print(f"Evaluation outputs: {output_dir}")
    return per_query, per_class, summary


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
        default=Path(__file__).resolve().parent / "results",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=6,
        help="Number of top-ranked neighbors used for precision/recall/F1.",
    )
    args = parser.parse_args()
    evaluate_top_k(
        args.database,
        args.features,
        args.output_dir,
        top_k=args.top_k,
    )


if __name__ == "__main__":
    main()
