"""Visualize current weighted mesh features with an interactive t-SNE plot."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.manifold import TSNE

from query import DESCRIPTOR_WEIGHTS
from scalability import FeatureSearchIndex


PROJECT_DIR = Path(__file__).resolve().parent


def create_embedding(
    database_path,
    features_path,
    perplexity=30.0,
    max_iter=1000,
    seed=42,
):
    """Create 2-D coordinates from the same weighted features used in search."""
    index = FeatureSearchIndex(
        database_path, features_path, DESCRIPTOR_WEIGHTS
    )
    feature_vectors = index.feature_vectors
    if perplexity <= 0 or perplexity >= len(feature_vectors):
        raise ValueError(
            "Perplexity must be positive and smaller than the candidate count"
        )
    if max_iter < 250:
        raise ValueError("t-SNE requires at least 250 iterations")

    model = TSNE(
        n_components=2,
        perplexity=perplexity,
        max_iter=max_iter,
        n_iter_without_progress=300,
        random_state=seed,
        init="pca",
        learning_rate="auto",
        metric="manhattan",
        n_jobs=-1,
    )
    coordinates = model.fit_transform(feature_vectors)
    embedding = index.candidates[["Class", "File"]].copy()
    embedding["TSNE_1"] = coordinates[:, 0]
    embedding["TSNE_2"] = coordinates[:, 1]
    return embedding


def plot_embedding(embedding, output_path, show=True):
    """Save class-colored coordinates and optionally show hover details."""
    classes = embedding["Class"].to_numpy()
    filenames = embedding["File"].to_numpy()
    coordinates = embedding[["TSNE_1", "TSNE_2"]].to_numpy(dtype=float)
    unique_classes, class_ids = np.unique(classes, return_inverse=True)

    figure, axis = plt.subplots(figsize=(16, 10))
    scatter = axis.scatter(
        coordinates[:, 0],
        coordinates[:, 1],
        c=class_ids,
        cmap="nipy_spectral",
        s=24,
        alpha=0.8,
    )
    highlight = axis.scatter(
        [],
        [],
        s=48,
        facecolors="none",
        edgecolors="black",
        linewidths=0.8,
        zorder=3,
    )
    annotation = axis.annotate(
        "",
        xy=(0, 0),
        xytext=(12, 12),
        textcoords="offset points",
        bbox={"boxstyle": "round", "fc": "white", "alpha": 0.9},
    )
    annotation.set_visible(False)

    legend_handles = [
        Line2D(
            [],
            [],
            marker="o",
            linestyle="",
            color=scatter.cmap(scatter.norm(class_id)),
            label=category,
        )
        for class_id, category in enumerate(unique_classes)
    ]
    axis.legend(
        handles=legend_handles,
        title="Shape classes",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.12),
        fontsize=8,
        ncol=min(8, len(unique_classes)),
    )
    axis.set_title("t-SNE of weighted shape features (Manhattan metric)")
    axis.set_xlabel("t-SNE dimension 1")
    axis.set_ylabel("t-SNE dimension 2")
    figure.subplots_adjust(bottom=0.28)

    selected_class = None

    def on_hover(event):
        nonlocal selected_class
        if event.inaxes == axis:
            contains, info = scatter.contains(event)
        else:
            contains, info = False, {}
        if contains and len(info["ind"]):
            point_index = int(info["ind"][0])
            category = classes[point_index]
            annotation.xy = coordinates[point_index]
            annotation.set_text(
                f"Class: {category}\nFile: {filenames[point_index]}"
            )
            annotation.set_visible(True)
            if selected_class != category:
                selected_class = category
                mask = classes == category
                scatter.set_alpha(0.12)
                highlight.set_offsets(coordinates[mask])
                class_color = scatter.cmap(
                    scatter.norm(class_ids[point_index])
                )
                highlight.set_edgecolors([class_color])
        elif selected_class is not None:
            selected_class = None
            scatter.set_alpha(0.8)
            highlight.set_offsets(np.empty((0, 2)))
            annotation.set_visible(False)
        figure.canvas.draw_idle()

    figure.canvas.mpl_connect("motion_notify_event", on_hover)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    else:
        plt.close(figure)
    return output_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--database",
        type=Path,
        default=PROJECT_DIR / "shapes" / "ShapeDatabase_INFOMR_cleaned",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=PROJECT_DIR / "features",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR / "visualizations",
    )
    parser.add_argument("--perplexity", type=float, default=30.0)
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Save the plot without opening an interactive window.",
    )
    args = parser.parse_args()

    embedding = create_embedding(
        args.database,
        args.features,
        perplexity=args.perplexity,
        max_iter=args.max_iter,
        seed=args.seed,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embedding_path = args.output_dir / "tsne_embedding.csv"
    plot_path = args.output_dir / "tsne_classes.png"
    embedding.to_csv(embedding_path, index=False)
    plot_embedding(embedding, plot_path, show=not args.no_show)
    print(f"Meshes visualized: {len(embedding)}")
    print(f"Classes: {embedding['Class'].nunique()}")
    print(f"Saved coordinates: {embedding_path}")
    print(f"Saved plot: {plot_path}")


if __name__ == "__main__":
    main()
