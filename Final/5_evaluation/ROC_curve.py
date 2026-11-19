"""Generate ROC and precision-recall evaluation for current mesh features."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from evaluation_common import (
    FINAL_DIR,
    load_ranked_feature_distances,
    precision_recall_for_query,
    query_relevance,
    roc_curve_for_query,
)


def evaluate_ranking(
    database_path,
    features_path,
    output_dir,
    query_file=None,
    query_class=None,
):
    """Compute full-gallery macro curves and per-query/category metrics."""
    index, distances, _ = load_ranked_feature_distances(
        database_path, features_path
    )
    candidates = index.candidates
    classes = candidates["Class"].to_numpy(dtype=str)
    filenames = candidates["File"].to_numpy(dtype=str)
    query_indices = np.arange(len(candidates))
    if query_file is not None:
        selected = filenames == query_file
        if query_class is not None:
            selected &= classes == query_class
        query_indices = np.flatnonzero(selected)
        if not len(query_indices):
            qualifier = (
                f" in class {query_class!r}" if query_class else ""
            )
            raise ValueError(
                f"No usable query mesh named {query_file!r}{qualifier}"
            )
    elif query_class is not None:
        query_indices = np.flatnonzero(classes == query_class)
        if not len(query_indices):
            raise ValueError(f"No usable query meshes in class {query_class!r}")

    metrics = []
    for query_index in query_indices:
        relevant = query_relevance(classes, int(query_index))
        if not relevant.any() or relevant.all():
            continue
        _, _, auc = roc_curve_for_query(
            distances[query_index], relevant
        )
        _, _, average_precision = precision_recall_for_query(
            distances[query_index], relevant
        )
        metrics.append({
            "Query_Class": classes[query_index],
            "Query_File": filenames[query_index],
            "AUC": auc,
            "Average_Precision": average_precision,
            "Relevant_Count": int(relevant.sum()),
            "Candidate_Count": len(relevant),
            "Positive_Prevalence": float(relevant.mean()),
        })
    if not metrics:
        raise ValueError("No queries have both relevant and irrelevant meshes")

    auc_df = pd.DataFrame(metrics)
    fpr_grid = np.linspace(0, 1, 1001)
    recall_grid = np.linspace(0, 1, 1001)
    mean_tpr = np.zeros_like(fpr_grid)
    mean_precision = np.zeros_like(recall_grid)
    for query_index in query_indices:
        relevant = query_relevance(classes, int(query_index))
        if not relevant.any() or relevant.all():
            continue
        fpr, tpr, _ = roc_curve_for_query(
            distances[query_index], relevant
        )
        recall, precision, _ = precision_recall_for_query(
            distances[query_index], relevant
        )
        mean_tpr += np.interp(fpr_grid, fpr, tpr)
        mean_precision += np.interp(recall_grid, recall, precision)
    mean_tpr /= len(auc_df)
    mean_precision /= len(auc_df)

    category_df = auc_df.groupby("Query_Class", sort=True).agg(
        Query_Count=("Query_File", "count"),
        Mean_AUC=("AUC", "mean"),
        Median_AUC=("AUC", "median"),
        Mean_Average_Precision=("Average_Precision", "mean"),
        Median_Average_Precision=("Average_Precision", "median"),
    ).reset_index()

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    auc_df.to_csv(output_dir / "roc_auc_by_query.csv", index=False)
    category_df.to_csv(output_dir / "roc_auc_by_category.csv", index=False)
    pd.DataFrame({
        "FPR": fpr_grid,
        "Mean_TPR": mean_tpr,
    }).to_csv(output_dir / "roc_curve_data.csv", index=False)
    pd.DataFrame({
        "Recall": recall_grid,
        "Mean_Precision": mean_precision,
    }).to_csv(
        output_dir / "precision_recall_curve_data.csv", index=False
    )

    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(
        fpr_grid,
        mean_tpr,
        label=f"Weighted feature score (mean AUC = {auc_df['AUC'].mean():.3f})",
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="False positive rate",
        ylabel="True positive rate",
        title="Macro-averaged ROC curve",
    )
    axis.grid(alpha=0.3)
    axis.legend(loc="lower right")
    figure.tight_layout()
    figure.savefig(output_dir / "roc_curve.png", dpi=200)
    plt.close(figure)

    mean_prevalence = float(auc_df["Positive_Prevalence"].mean())
    figure, axis = plt.subplots(figsize=(8, 6))
    axis.plot(
        recall_grid,
        mean_precision,
        label=(
            "Weighted feature score "
            f"(mean AP = {auc_df['Average_Precision'].mean():.3f})"
        ),
    )
    axis.axhline(
        mean_prevalence,
        color="gray",
        linestyle=":",
        label=f"Mean positive prevalence = {mean_prevalence:.3f}",
    )
    axis.set(
        xlim=(0, 1),
        ylim=(0, 1.02),
        xlabel="Recall",
        ylabel="Precision",
        title="Macro-averaged precision-recall curve",
    )
    axis.grid(alpha=0.3)
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(
        output_dir / "precision_recall_curve.png",
        dpi=200,
    )
    plt.close(figure)

    category_order = category_df.sort_values(
        ["Mean_AUC", "Query_Class"], kind="stable"
    )
    figure, axis = plt.subplots(
        figsize=(10, max(8, len(category_order) * 0.22))
    )
    axis.barh(
        category_order["Query_Class"],
        category_order["Mean_AUC"],
        color="#3977A8",
    )
    axis.axvline(0.5, color="gray", linestyle="--", linewidth=1)
    axis.set(
        xlim=(0, 1),
        xlabel="Mean per-query ROC AUC",
        ylabel="Query class",
        title="ROC AUC by class",
    )
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(
        output_dir / "roc_auc_by_category.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    ordered = auc_df.sort_values(
        ["AUC", "Query_Class", "Query_File"], kind="stable"
    )
    best = ordered.iloc[-1]
    worst = ordered.iloc[0]
    figure, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True, sharey=True)
    for axis, row, label in (
        (axes[0], best, "Best"),
        (axes[1], worst, "Worst"),
    ):
        query_index = int(np.flatnonzero(
            (classes == row["Query_Class"])
            & (filenames == row["Query_File"])
        )[0])
        relevant = query_relevance(classes, query_index)
        fpr, tpr, _ = roc_curve_for_query(
            distances[query_index], relevant
        )
        axis.plot(fpr, tpr, label=f"AUC = {row['AUC']:.3f}")
        axis.plot([0, 1], [0, 1], linestyle=":", color="gray")
        axis.set(
            xlim=(0, 1),
            ylim=(0, 1.02),
            xlabel="False positive rate",
            title=(
                f"{label}: {row['Query_Class']}/{row['Query_File']}\n"
                f"AUC = {row['AUC']:.3f}"
            ),
        )
        axis.grid(alpha=0.25)
        axis.legend(loc="lower right")
    axes[0].set_ylabel("True positive rate")
    figure.tight_layout()
    figure.savefig(
        output_dir / "roc_auc_best_worst_queries.png",
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(figure)

    print(f"Usable candidates: {len(candidates)}")
    print(f"Queries evaluated: {len(auc_df)}")
    print(f"Mean ROC AUC: {auc_df['AUC'].mean():.4f}")
    print(f"Mean average precision: {auc_df['Average_Precision'].mean():.4f}")
    print(
        f"Best query: {best['Query_Class']}/{best['Query_File']} "
        f"(AUC={best['AUC']:.4f})"
    )
    print(
        f"Worst query: {worst['Query_Class']}/{worst['Query_File']} "
        f"(AUC={worst['AUC']:.4f})"
    )
    print(f"ROC and PR outputs: {output_dir}")
    return auc_df, category_df


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
        "--query",
        help="Evaluate only this mesh filename (optional).",
    )
    parser.add_argument(
        "--query-class",
        help="Disambiguate a query filename shared by multiple classes.",
    )
    args = parser.parse_args()
    evaluate_ranking(
        args.database,
        args.features,
        args.output_dir,
        query_file=args.query,
        query_class=args.query_class,
    )


if __name__ == "__main__":
    main()
