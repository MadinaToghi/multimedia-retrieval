import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D


PROJECT_DIR = Path(__file__).resolve().parent
ROC_DIR = PROJECT_DIR / "roc"
RESULT_PATHS = {
    "KNN": ROC_DIR / "knn_all_distances.csv",
    "RNN": ROC_DIR / "rnn_all_distances.csv",
}
ROC_PLOT_PATH = ROC_DIR / "roc_curve.png"
ROC_DATA_PATH = ROC_DIR / "roc_curve_data.csv"
AUC_DATA_PATH = ROC_DIR / "roc_auc_by_query.csv"
CATEGORY_AUC_PATH = ROC_DIR / "roc_auc_by_category.csv"
CATEGORY_PLOT_PATH = ROC_DIR / "roc_auc_by_category.png"
EXTREMES_PLOT_PATH = ROC_DIR / "roc_auc_best_worst_queries.png"


def roc_curve_for_query(
    distances: np.ndarray,
    relevant: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    positive_count = int(relevant.sum())
    negative_count = len(relevant) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("A ROC curve requires both relevant and irrelevant meshes.")

    order = np.argsort(distances, kind="stable")
    sorted_distances = distances[order]
    sorted_relevant = relevant[order].astype(int)
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_distances)) + 1, len(order)]

    true_positives = np.cumsum(sorted_relevant)[group_ends - 1]
    false_positives = group_ends - true_positives
    tpr = np.r_[0.0, true_positives / positive_count]
    fpr = np.r_[0.0, false_positives / negative_count]
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    return fpr, tpr, auc


def precision_recall_for_query(
    distances: np.ndarray,
    relevant: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    positive_count = int(relevant.sum())
    if positive_count == 0:
        raise ValueError("A precision-recall curve requires relevant meshes.")

    order = np.argsort(distances, kind="stable")
    sorted_distances = distances[order]
    sorted_relevant = relevant[order].astype(int)
    group_ends = np.r_[np.flatnonzero(np.diff(sorted_distances)) + 1, len(order)]

    true_positives = np.cumsum(sorted_relevant)[group_ends - 1]
    predicted_positives = group_ends
    recall = np.r_[0.0, true_positives / positive_count]
    precision = np.r_[1.0, true_positives / predicted_positives]
    average_precision = float(
        np.sum(np.diff(recall) * precision[1:])
    )
    return recall, precision, average_precision


def summarize_categories(auc_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method, query_class), group in auc_df.groupby(
        ["Method", "Query_Class"], sort=True
    ):
        ordered = group.sort_values(["AUC", "Query_File"], kind="stable")
        worst = ordered.iloc[0]
        best = ordered.iloc[-1]
        rows.append({
            "Method": method,
            "Query_Class": query_class,
            "Query_Count": len(group),
            "Mean_AUC": group["AUC"].mean(),
            "Median_AUC": group["AUC"].median(),
            "Mean_Average_Precision": group["Average_Precision"].mean(),
            "Median_Average_Precision": group["Average_Precision"].median(),
            "Best_Query_File": best["Query_File"],
            "Best_Query_AUC": best["AUC"],
            "Worst_Query_File": worst["Query_File"],
            "Worst_Query_AUC": worst["AUC"],
        })
    return pd.DataFrame(rows)


def plot_category_auc(auc_df: pd.DataFrame, category_df: pd.DataFrame) -> None:
    category_order = (
        category_df[category_df["Method"] == "KNN"]
        .sort_values(["Mean_AUC", "Query_Class"])["Query_Class"]
        .tolist()
    )
    positions = np.arange(1, len(category_order) + 1)
    colors = {"KNN": "#3977A8", "RNN": "#E68632"}
    fig, ax = plt.subplots(figsize=(12, max(14, len(category_order) * 0.23)))
    rng = np.random.default_rng(42)

    for method, offset in (("KNN", -0.18), ("RNN", 0.18)):
        grouped_values = [
            auc_df.loc[
                (auc_df["Method"] == method)
                & (auc_df["Query_Class"] == query_class),
                "AUC",
            ].to_numpy()
            for query_class in category_order
        ]
        boxplot = ax.boxplot(
            grouped_values,
            positions=positions + offset,
            orientation="horizontal",
            widths=0.32,
            patch_artist=True,
            showmeans=True,
            meanprops={
                "marker": "D",
                "markerfacecolor": "white",
                "markeredgecolor": colors[method],
                "markersize": 4,
            },
            medianprops={"color": "black", "linewidth": 1.2},
            flierprops={
                "marker": ".",
                "markersize": 3,
                "markerfacecolor": colors[method],
                "markeredgecolor": colors[method],
                "alpha": 0.45,
            },
        )
        for box in boxplot["boxes"]:
            box.set_facecolor(colors[method])
            box.set_alpha(0.35)

        for position, values in zip(positions + offset, grouped_values):
            jitter = rng.normal(0, 0.035, len(values))
            ax.scatter(
                values,
                position + jitter,
                s=10,
                color=colors[method],
                alpha=0.4,
                linewidths=0,
            )

    ax.set_yticks(positions, labels=category_order, fontsize=7)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Per-query ROC AUC")
    ax.set_ylabel("Query category")
    ax.set_title("Per-category ROC AUC distributions")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Random ranking")
    ax.grid(axis="x", alpha=0.25)
    ax.legend(
        handles=[
            Patch(facecolor=colors["KNN"], alpha=0.45, label="KNN"),
            Patch(facecolor=colors["RNN"], alpha=0.45, label="RNN"),
            Line2D([0], [0], color="black", marker="_", linestyle="None",
                   markersize=12, label="Median"),
            Line2D([0], [0], color=colors["KNN"], marker="D",
                   markerfacecolor="white", linestyle="None", label="Mean"),
        ],
        loc="lower right",
    )
    fig.tight_layout()
    fig.savefig(CATEGORY_PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_extreme_queries(
    auc_df: pd.DataFrame,
    query_curves: dict,
) -> tuple[pd.Series, pd.Series]:
    knn_auc = auc_df[auc_df["Method"] == "KNN"]
    ordered = knn_auc.sort_values(["AUC", "Query_File"], kind="stable")
    worst = ordered.iloc[0]
    best = ordered.iloc[-1]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True, sharey=True)
    for ax, row, label, color in (
        (axes[0], best, "Best", "#25855A"),
        (axes[1], worst, "Worst", "#C64B46"),
    ):
        key = (row["Query_File"], row["Query_Class"])
        for method, linestyle in (("KNN", "-"), ("RNN", "--")):
            fpr, tpr = query_curves[method][key]
            method_auc = auc_df.loc[
                (auc_df["Method"] == method)
                & (auc_df["Query_File"] == row["Query_File"]),
                "AUC",
            ].iloc[0]
            ax.plot(
                fpr,
                tpr,
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                label=f"{method} AUC = {method_auc:.3f}",
            )
        ax.plot([0, 1], [0, 1], color="gray", linestyle=":", label="Random ranking")
        ax.set_title(
            f"{label}: {row['Query_File']} ({row['Query_Class']})\n"
            f"AUC = {row['AUC']:.3f}"
        )
        ax.set_xlabel("False Positive Rate")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.25)
        ax.legend(loc="lower right")

    axes[0].set_ylabel("True Positive Rate")
    fig.suptitle("Best and worst single-query ROC curves")
    fig.tight_layout()
    fig.savefig(EXTREMES_PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return best, worst


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate separate KNN and RNN ROC curves using full-gallery "
            "distance outputs."
        )
    )
    parser.add_argument(
        "--query",
        help="Generate curves for this query mesh only.",
    )
    args = parser.parse_args()

    ROC_DIR.mkdir(parents=True, exist_ok=True)

    method_curves = {}
    all_query_curves = {}
    all_query_pr_curves = {}
    auc_rows = []
    fpr_grid = np.linspace(0.0, 1.0, 1001)
    recall_grid = np.linspace(0.0, 1.0, 1001)
    for method, result_path in RESULT_PATHS.items():
        results = pd.read_csv(result_path)
        required_columns = {
            "Query_File", "Query_Class", "File", "Class", "Distance"
        }
        if not required_columns.issubset(results.columns):
            raise ValueError(
                f"{result_path.name} must contain {sorted(required_columns)}."
            )
        if results.empty:
            raise ValueError(f"No query results found in {result_path.name}.")

        all_query_curves[method] = {}
        all_query_pr_curves[method] = {}
        for (query_file, query_class), query_results in results.groupby(
            ["Query_File", "Query_Class"], sort=False
        ):
            relevant = query_results["Class"].to_numpy() == query_class
            distances = query_results["Distance"].to_numpy(dtype=float)
            fpr, tpr, auc = roc_curve_for_query(distances, relevant)
            recall, precision, average_precision = precision_recall_for_query(
                distances,
                relevant,
            )
            all_query_curves[method][(query_file, query_class)] = (fpr, tpr)
            all_query_pr_curves[method][(query_file, query_class)] = (
                recall,
                precision,
            )
            auc_rows.append({
                "Method": method,
                "Query_File": query_file,
                "Query_Class": query_class,
                "AUC": auc,
                "Average_Precision": average_precision,
                "Relevant_Count": int(relevant.sum()),
                "Candidate_Count": len(relevant),
                "Positive_Prevalence": float(relevant.mean()),
            })

    auc_df = pd.DataFrame(auc_rows)
    query_sets = {
        method: set(
            zip(
                auc_df.loc[auc_df["Method"] == method, "Query_File"],
                auc_df.loc[auc_df["Method"] == method, "Query_Class"],
            )
        )
        for method in RESULT_PATHS
    }
    if query_sets["KNN"] != query_sets["RNN"]:
        raise ValueError("KNN and RNN outputs do not contain the same queries.")

    if set(all_query_curves) != {"KNN", "RNN"}:
        raise ValueError("Both KNN and RNN results are required.")

    if set(all_query_pr_curves) != {"KNN", "RNN"}:
        raise ValueError("Both KNN and RNN precision-recall results are required.")

    if args.query:
        selected_keys = {
            key for key in query_sets["KNN"] if key[0] == args.query
        }
        if not selected_keys:
            raise ValueError(f"No query named {args.query!r} found in the results.")
        selected_query_curves = {
            method: [all_query_curves[method][key] for key in selected_keys]
            for method in RESULT_PATHS
        }
        selected_pr_curves = {
            method: [all_query_pr_curves[method][key] for key in selected_keys]
            for method in RESULT_PATHS
        }
    else:
        selected_query_curves = {
            method: list(all_query_curves[method].values())
            for method in RESULT_PATHS
        }
        selected_pr_curves = {
            method: list(all_query_pr_curves[method].values())
            for method in RESULT_PATHS
        }

    method_curves = {
        method: np.mean(
            [np.interp(fpr_grid, curve_fpr, curve_tpr)
             for curve_fpr, curve_tpr in curves],
            axis=0,
        )
        for method, curves in selected_query_curves.items()
    }
    method_pr_curves = {
        method: np.mean(
            [np.interp(recall_grid, curve_recall, curve_precision)
             for curve_recall, curve_precision in curves],
            axis=0,
        )
        for method, curves in selected_pr_curves.items()
    }

    plot_fpr = np.r_[0.0, fpr_grid]
    plot_curves = {
        method: np.r_[0.0, mean_tpr]
        for method, mean_tpr in method_curves.items()
    }
    curve_data = pd.DataFrame({
        "FPR": plot_fpr,
        "KNN_Mean_TPR": plot_curves["KNN"],
        "RNN_Mean_TPR": plot_curves["RNN"],
    })
    curve_data.to_csv(ROC_DATA_PATH, index=False)

    plot_recall = np.r_[0.0, recall_grid]
    plot_pr_curves = {
        method: np.r_[1.0, mean_precision]
        for method, mean_precision in method_pr_curves.items()
    }
    precision_recall_data = pd.DataFrame({
        "Recall": plot_recall,
        "KNN_Mean_Precision": plot_pr_curves["KNN"],
        "RNN_Mean_Precision": plot_pr_curves["RNN"],
    })
    precision_recall_data.to_csv(
        ROC_DIR / "precision_recall_curve_data.csv",
        index=False,
    )
    auc_df.to_csv(AUC_DATA_PATH, index=False)

    category_df = summarize_categories(auc_df)
    category_df.to_csv(CATEGORY_AUC_PATH, index=False)
    plot_category_auc(auc_df, category_df)
    best_query, worst_query = plot_extreme_queries(auc_df, all_query_curves)

    plt.figure(figsize=(8, 6))
    for method in RESULT_PATHS:
        mean_auc = auc_df.loc[auc_df["Method"] == method, "AUC"].mean()
        plt.plot(
            plot_fpr,
            plot_curves[method],
            label=f"{method} (mean AUC = {mean_auc:.3f})",
            linestyle="-" if method == "KNN" else "--",
        )
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random ranking")
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(
        f"{'ROC Curves for ' + args.query if args.query else 'Macro-Averaged ROC Curves'}"
    )
    plt.grid(alpha=0.3)
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(ROC_PLOT_PATH, dpi=200, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(8, 6))
    for method in RESULT_PATHS:
        selected_auc = auc_df[auc_df["Method"] == method]
        if args.query:
            selected_auc = selected_auc[selected_auc["Query_File"] == args.query]
        mean_ap = selected_auc["Average_Precision"].mean()
        plt.plot(
            plot_recall,
            plot_pr_curves[method],
            label=f"{method} (mean AP = {mean_ap:.3f})",
            linestyle="-" if method == "KNN" else "--",
        )
    selected_prevalence = auc_df["Positive_Prevalence"]
    if args.query:
        selected_prevalence = auc_df.loc[
            auc_df["Query_File"] == args.query, "Positive_Prevalence"
        ]
    plt.axhline(
        selected_prevalence.mean(),
        color="gray",
        linestyle=":",
        label=f"Positive prevalence = {selected_prevalence.mean():.3f}",
    )
    plt.xlim(0, 1)
    plt.ylim(0, 1.02)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(
        f"{'Precision-Recall Curves for ' + args.query if args.query else 'Macro-Averaged Precision-Recall Curves'}"
    )
    plt.grid(alpha=0.3)
    plt.legend(loc="upper right")
    plt.tight_layout()
    precision_recall_plot_path = ROC_DIR / "precision_recall_curve.png"
    plt.savefig(precision_recall_plot_path, dpi=200, bbox_inches="tight")
    plt.close()

    print(f"Queries evaluated per method: {len(query_sets['KNN'])}")
    for method in RESULT_PATHS:
        method_auc = auc_df[auc_df["Method"] == method]
        mean_auc = method_auc["AUC"].mean()
        median_auc = method_auc["AUC"].median()
        mean_ap = method_auc["Average_Precision"].mean()
        median_ap = method_auc["Average_Precision"].median()
        method_categories = category_df[category_df["Method"] == method]
        best_category = method_categories.sort_values(
            ["Mean_AUC", "Query_Class"], kind="stable"
        ).iloc[-1]
        worst_category = method_categories.sort_values(
            ["Mean_AUC", "Query_Class"], kind="stable"
        ).iloc[0]
        print(
            f"{method}: mean AUC={mean_auc:.4f}, median AUC={median_auc:.4f}, "
            f"mean AP={mean_ap:.4f}, median AP={median_ap:.4f}"
        )
        print(
            f"  Best category by mean AUC: {best_category['Query_Class']} "
            f"({best_category['Mean_AUC']:.4f})"
        )
        print(
            f"  Worst category by mean AUC: {worst_category['Query_Class']} "
            f"({worst_category['Mean_AUC']:.4f})"
        )
        print(
            f"  Best query: {best_query['Query_File']} "
            f"({best_query['Query_Class']}, AUC={best_query['AUC']:.4f})"
        )
        print(
            f"  Worst query: {worst_query['Query_File']} "
            f"({worst_query['Query_Class']}, AUC={worst_query['AUC']:.4f})"
        )
    print(f"ROC curve: {ROC_PLOT_PATH.relative_to(PROJECT_DIR)}")
    print(f"ROC data: {ROC_DATA_PATH.relative_to(PROJECT_DIR)}")
    print(f"Per-query AUCs: {AUC_DATA_PATH.relative_to(PROJECT_DIR)}")
    print(f"Per-category AUCs: {CATEGORY_AUC_PATH.relative_to(PROJECT_DIR)}")
    print(f"Category plot: {CATEGORY_PLOT_PATH.relative_to(PROJECT_DIR)}")
    print(f"Best/worst query plot: {EXTREMES_PLOT_PATH.relative_to(PROJECT_DIR)}")
    print(
        "Precision-recall plot: "
        f"{precision_recall_plot_path.relative_to(PROJECT_DIR)}"
    )
    print(
        "Precision-recall data: "
        f"{(ROC_DIR / 'precision_recall_curve_data.csv').relative_to(PROJECT_DIR)}"
    )


if __name__ == "__main__":
    main()
