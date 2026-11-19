"""Explore mesh counts and shape-database structure."""

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

import MMR_utility as mmru


DEFAULT_SOURCE = PROJECT_DIR / "shapes" / "ShapeDatabase_INFOMR"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results"


def read_obj_face_info(obj_path):
    """Count OBJ polygon faces and classify triangles, quads, or mixed faces."""
    face_count = 0
    face_sizes = set()
    with Path(obj_path).open(encoding="utf-8", errors="replace") as obj_file:
        for line in obj_file:
            parts = line.split()
            if parts and parts[0] == "f":
                face_count += 1
                face_sizes.add(len(parts) - 1)

    if face_sizes == {3}:
        face_type = "triangles"
    elif face_sizes == {4}:
        face_type = "quads"
    elif face_sizes:
        face_type = "mixed"
    else:
        face_type = "unknown"
    return face_count, face_type


def analyze_mesh(mesh_path):
    """Collect mesh size, source face type, and bounding-box statistics."""
    mesh_path = Path(mesh_path)
    mesh = mmru.load_mesh(mesh_path)
    face_count, face_type = read_obj_face_info(mesh_path)
    bounds = np.asarray(mesh.bounds, dtype=np.float64)
    if bounds.shape != (2, 3) or not np.all(np.isfinite(bounds)):
        raise ValueError(f"Mesh has invalid bounds: {mesh_path}")
    return {
        "Class": mesh_path.parent.name,
        "File": mesh_path.name,
        "Vertices Count": len(mesh.vertices),
        "Faces Count": face_count,
        "Face Type": face_type,
        "Bounds Min": json.dumps(bounds[0].tolist()),
        "Bounds Max": json.dumps(bounds[1].tolist()),
    }


def find_representative_shapes(mesh_statistics):
    """Find average-like and minimum/maximum vertex- and face-count meshes."""
    if mesh_statistics.empty:
        raise ValueError("No successfully analyzed meshes are available")

    frame = mesh_statistics.copy()
    mean_vertices = float(frame["Vertices Count"].mean())
    mean_faces = float(frame["Faces Count"].mean())
    vertex_scale = max(abs(mean_vertices), 1.0)
    face_scale = max(abs(mean_faces), 1.0)
    average_distance = (
        (frame["Vertices Count"] - mean_vertices).abs() / vertex_scale
        + (frame["Faces Count"] - mean_faces).abs() / face_scale
    )
    representatives = [
        ("Average-like shape", frame.loc[average_distance.idxmin()]),
        ("Fewest vertices", frame.loc[frame["Vertices Count"].idxmin()]),
        ("Most vertices", frame.loc[frame["Vertices Count"].idxmax()]),
        ("Fewest faces", frame.loc[frame["Faces Count"].idxmin()]),
        ("Most faces", frame.loc[frame["Faces Count"].idxmax()]),
    ]
    return pd.DataFrame([
        {
            "Representative": label,
            "Class": row["Class"],
            "File": row["File"],
            "Vertices Count": int(row["Vertices Count"]),
            "Faces Count": int(row["Faces Count"]),
        }
        for label, row in representatives
    ]).drop_duplicates(["Class", "File"], keep="first")


def summarize_meshes(mesh_statistics):
    """Summarize mesh counts and vertex/face statistics by category."""
    rows = []
    for category, group in mesh_statistics.groupby("Class", sort=True):
        face_type_counts = group["Face Type"].value_counts()
        rows.append({
            "Class": category,
            "Mesh Count": len(group),
            "Mean Vertices": group["Vertices Count"].mean(),
            "Median Vertices": group["Vertices Count"].median(),
            "Mean Faces": group["Faces Count"].mean(),
            "Median Faces": group["Faces Count"].median(),
            "Triangle Meshes": int(face_type_counts.get("triangles", 0)),
            "Quad Meshes": int(face_type_counts.get("quads", 0)),
            "Mixed Meshes": int(face_type_counts.get("mixed", 0)),
            "Unknown Face Type": int(face_type_counts.get("unknown", 0)),
        })
    return pd.DataFrame(rows)


def plot_statistics(mesh_statistics, summary_by_class, output_dir, show=False):
    """Save distributions for vertices, faces, and category size."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    figure, axes = plt.subplots(2, 1, figsize=(12, 9))
    mesh_statistics["Vertices Count"].plot.hist(
        bins=30, ax=axes[0], color="steelblue", edgecolor="white"
    )
    axes[0].set(
        title="Vertex count distribution",
        xlabel="Vertices per mesh",
        ylabel="Mesh count",
    )
    mesh_statistics["Faces Count"].plot.hist(
        bins=30, ax=axes[1], color="darkorange", edgecolor="white"
    )
    axes[1].set(
        title="Face count distribution",
        xlabel="Faces per mesh (OBJ polygons)",
        ylabel="Mesh count",
    )
    figure.tight_layout()
    distributions_path = output_dir / "database_distributions.png"
    figure.savefig(distributions_path, dpi=200, bbox_inches="tight")

    class_sizes = summary_by_class.sort_values(
        ["Mesh Count", "Class"], ascending=[True, True]
    )
    class_figure, class_axis = plt.subplots(
        figsize=(10, max(10, len(class_sizes) * 0.25))
    )
    class_axis.barh(
        class_sizes["Class"],
        class_sizes["Mesh Count"],
        color="seagreen",
    )
    class_axis.tick_params(axis="y", labelsize=7)
    class_axis.set(
        title="Meshes per class",
        xlabel="Mesh count",
        ylabel="Class",
    )
    class_figure.tight_layout()
    class_sizes_path = output_dir / "class_sizes.png"
    class_figure.savefig(class_sizes_path, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(figure)
    plt.close(class_figure)
    return distributions_path, class_sizes_path


def explore_database(
    source_path,
    output_dir,
    sample_per_category=10,
    full_database=True,
    seed=43,
    show=False,
):
    """Analyze the source database or a reproducible per-class sample."""
    source_path = Path(source_path).resolve()
    output_dir = Path(output_dir).resolve()
    all_meshes = mmru.find_obj_files(source_path)
    if not all_meshes:
        raise ValueError(f"No OBJ meshes found under {source_path}")
    if full_database:
        selected_meshes = all_meshes
    else:
        if sample_per_category < 1:
            raise ValueError("Sample size per category must be at least one")
        selected_meshes = mmru.stratified_mesh_sample(
            all_meshes, sample_per_category, seed
        )

    rows = []
    for index, mesh_path in enumerate(selected_meshes, start=1):
        try:
            row = analyze_mesh(mesh_path)
            row["Status"] = "OK"
            row["Error"] = ""
        except Exception as error:
            row = {
                "Class": mesh_path.parent.name,
                "File": mesh_path.name,
                "Vertices Count": np.nan,
                "Faces Count": np.nan,
                "Face Type": "unknown",
                "Bounds Min": "",
                "Bounds Max": "",
                "Status": "ERROR",
                "Error": str(error),
            }
        rows.append(row)
        if index % 100 == 0 or index == len(selected_meshes):
            print(f"Analyzed {index}/{len(selected_meshes)} meshes")

    output_dir.mkdir(parents=True, exist_ok=True)
    statistics = pd.DataFrame(rows)
    statistics.to_csv(output_dir / "mesh_statistics.csv", index=False)
    failures = statistics.loc[statistics["Status"] == "ERROR"]
    if not failures.empty:
        failures.to_csv(output_dir / "mesh_statistics_errors.csv", index=False)

    successful = statistics.loc[statistics["Status"] == "OK"].copy()
    if successful.empty:
        raise ValueError(
            f"No meshes could be analyzed. See "
            f"{output_dir / 'mesh_statistics.csv'} for errors."
        )

    summary_by_class = summarize_meshes(successful)
    summary_by_class.to_csv(
        output_dir / "summary_by_class.csv", index=False
    )
    database_summary = pd.DataFrame([{
        "Source": str(source_path),
        "Sampled": not full_database,
        "Sample_Per_Category": (
            "all" if full_database else sample_per_category
        ),
        "Seed": seed,
        "Meshes_Selected": len(selected_meshes),
        "Meshes_Analyzed": len(successful),
        "Meshes_Failed": len(failures),
        "Class_Count": successful["Class"].nunique(),
        "Mean_Vertices": successful["Vertices Count"].mean(),
        "Mean_Faces": successful["Faces Count"].mean(),
    }])
    database_summary.to_csv(
        output_dir / "database_summary.csv", index=False
    )

    representatives = find_representative_shapes(successful)
    representatives.to_csv(
        output_dir / "representative_shapes.csv", index=False
    )
    plot_paths = plot_statistics(
        successful, summary_by_class, output_dir, show=show
    )

    print(f"Meshes analyzed successfully: {len(successful)}")
    print(f"Categories: {successful['Class'].nunique()}")
    print(f"Mean vertices: {successful['Vertices Count'].mean():.2f}")
    print(f"Mean faces: {successful['Faces Count'].mean():.2f}")
    if not failures.empty:
        print(
            f"Mesh analysis failures: {len(failures)} "
            f"(see {output_dir / 'mesh_statistics_errors.csv'})"
        )
    print("Representative meshes:")
    for _, representative in representatives.iterrows():
        print(
            f"  {representative['Representative']}: "
            f"{representative['Class']}/{representative['File']} "
            f"({representative['Vertices Count']} vertices, "
            f"{representative['Faces Count']} faces)"
        )
    print(f"Statistics and plots saved under: {output_dir}")
    for plot_path in plot_paths:
        print(f"Plot: {plot_path}")
    return statistics, summary_by_class, representatives


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="OBJ database root (default: Final/shapes/ShapeDatabase_INFOMR).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Where statistics CSVs and plots are written.",
    )
    parser.add_argument(
        "--sample-per-category",
        type=int,
        default=None,
        help="Optional stratified sample size; by default all meshes are analyzed.",
    )
    parser.add_argument(
        "--full",
        action="store_true",
        help="Analyze every OBJ instead of sampling per category.",
    )
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument(
        "--show",
        action="store_true",
        help="Open the plot window after saving the statistics plot.",
    )
    args = parser.parse_args()
    if args.sample_per_category is not None and args.sample_per_category < 1:
        parser.error("--sample-per-category must be at least one")
    full_database = args.full or args.sample_per_category is None
    explore_database(
        args.source,
        args.output_dir,
        sample_per_category=args.sample_per_category or 10,
        full_database=full_database,
        seed=args.seed,
        show=args.show,
    )


if __name__ == "__main__":
    main()
