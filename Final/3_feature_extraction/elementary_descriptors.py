"""Calculate raw elementary descriptors and database z-score statistics."""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "1_read_data"))

import MMR_utility as mmru


ELEMENTARY_COLUMNS = [
    "Volume", "Surface Area", "Compactness", "Rectangularity",
    "Diameter", "Convexity", "Eccentricity",
]
ELEMENTARY_VALIDITY_COLUMNS = {
    "Volume": "Volume_Valid",
    "Surface Area": "Surface_Area_Valid",
    "Compactness": "Compactness_Valid",
    "Rectangularity": "Rectangularity_Valid",
    "Diameter": "Diameter_Valid",
    "Convexity": "Convexity_Valid",
    "Eccentricity": "Eccentricity_Valid",
}
# Per-mesh feature calculations

def elementary_descriptors(mesh):
    """Calculate raw scalar mesh descriptors and diagnostic quality flags.

    Args:
        mesh: Preprocessed normalized Trimesh.
    Returns:
        Raw values, per-feature reliability flags, and a warning string. Flags
        are diagnostics; finite values remain available to the query scorer.
    """
    volume = abs(float(mesh.volume))
    area = float(mesh.area)
    obb_volume = float(mesh.bounding_box_oriented.volume)
    hull_volume = float(mesh.convex_hull.volume)
    eigenvalues = np.linalg.eigvalsh(np.cov(mesh.vertices, rowvar=False))
    smallest = float(eigenvalues[0])
    largest = float(eigenvalues[-1])

    hull_vertices = np.asarray(mesh.convex_hull.vertices)
    diameter = 0.0
    for start in range(0, len(hull_vertices), 512):
        block = hull_vertices[start:start + 512]
        diameter = max(
            diameter,
            float(np.linalg.norm(
                block[:, None, :] - hull_vertices[None, :, :], axis=2
            ).max()),
        )

    volume_reliable = bool(
        mesh.is_watertight
        and mesh.is_winding_consistent
        and np.isfinite(volume)
        and volume > 0
    )
    surface_area_valid = bool(np.isfinite(area) and area > 0)
    compactness = (
        36.0 * np.pi * volume**2 / area**3
        if surface_area_valid else np.nan
    )
    rectangularity = volume / obb_volume if obb_volume > 0 else np.nan
    convexity = volume / hull_volume if hull_volume > 0 else np.nan
    eccentricity = largest / smallest if smallest > 0 else np.nan
    tolerance = 1e-6
    valid = {
        "Volume": volume_reliable,
        "Surface Area": surface_area_valid,
        "Compactness": bool(
            volume_reliable and np.isfinite(compactness)
            and 0 <= compactness <= 1 + tolerance
        ),
        "Rectangularity": bool(
            volume_reliable and np.isfinite(rectangularity)
            and 0 <= rectangularity <= 1 + tolerance
        ),
        "Diameter": bool(np.isfinite(diameter) and diameter > 0),
        "Convexity": bool(
            volume_reliable and np.isfinite(convexity)
            and 0 <= convexity <= 1 + tolerance
        ),
        "Eccentricity": bool(
            np.isfinite(eccentricity) and eccentricity >= 1.0
        ),
    }
    result = {
        "Volume": volume,
        "Surface Area": area,
        "Compactness": compactness,
        "Rectangularity": rectangularity,
        "Diameter": diameter,
        "Convexity": convexity,
        "Eccentricity": eccentricity,
    }
    result.update({
        ELEMENTARY_VALIDITY_COLUMNS[name]: value
        for name, value in valid.items()
    })
    result["Quality_Warnings"] = "; ".join(
        f"{name} unreliable" for name, is_valid in valid.items() if not is_valid
    )
    return result


def standardize_elementary(values, scaler_frame):
    """Apply saved database z-score statistics to raw values.

    Args:
        values: Mapping from elementary feature names to raw values.
        scaler_frame: DataFrame containing Feature, Mean, and Std.
    Returns:
        Mapping of standardized values; only missing/non-finite values become
        NaN. Reliability flags do not suppress finite measurements.
    """
    scaler = scaler_frame.set_index("Feature")
    standardized = {}
    for feature in ELEMENTARY_COLUMNS:
        raw_value = values.get(feature, np.nan)
        if not np.isfinite(raw_value):
            standardized[feature] = np.nan
            continue
        standardized[feature] = (
            float(raw_value) - scaler.loc[feature, "Mean"]
        ) / scaler.loc[feature, "Std"]
    return standardized


# Database feature extraction and scaler fitting

def extract_elementary_descriptors(database_path):
    """Extract raw elementary values and warnings for every processed OBJ.

    Args:
        database_path: Root of the preprocessed database.
    Returns:
        DataFrame with class/file keys, raw values, validity flags, status, and
        any extraction error. Quality-warning rows are retained.
    """
    rows = []
    for mesh_path in tqdm(
        mmru.find_obj_files(database_path), desc="Elementary descriptors"
    ):
        row = {
            "Class": mesh_path.parent.name,
            "File": mesh_path.name,
            **{column: np.nan for column in ELEMENTARY_COLUMNS},
            **{column: False for column in ELEMENTARY_VALIDITY_COLUMNS.values()},
            "Quality_Warnings": "",
            "Status": "ERROR",
            "Error": "",
        }
        try:
            row.update(elementary_descriptors(mmru.load_mesh(mesh_path)))
            row["Status"] = (
                "OK" if all(
                    row[column]
                    for column in ELEMENTARY_VALIDITY_COLUMNS.values()
                ) else "WARNING"
            )
        except Exception as error:
            row["Error"] = str(error)
        rows.append(row)
    return pd.DataFrame(rows)


def elementary_scaler(descriptor_frame):
    """Fit population z-score statistics from every finite raw measurement.

    Args:
        descriptor_frame: Output of extract_elementary_descriptors.
    Returns:
        DataFrame containing Feature, Mean, and nonzero Std columns.
    """
    rows = []
    for feature in ELEMENTARY_COLUMNS:
        valid_rows = descriptor_frame["Status"].isin(["OK", "WARNING"])
        values = descriptor_frame.loc[valid_rows, feature].to_numpy(dtype=float)
        values = values[np.isfinite(values)]
        if not len(values):
            raise ValueError(f"No finite values for elementary descriptor {feature}")
        standard_deviation = float(values.std(ddof=0))
        rows.append({
            "Feature": feature,
            "Mean": float(values.mean()),
            "Std": standard_deviation if standard_deviation > 0 else 1.0,
        })
    return pd.DataFrame(rows)


def main():
    """Extract a database's elementary CSV and its reusable z-score scaler."""
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
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outputs = [
        args.features / "elementary_descriptors.csv",
        args.features / "elementary_scaler.csv",
    ]
    if not args.overwrite and any(path.exists() for path in outputs):
        parser.error(
            "Elementary outputs already exist; pass --overwrite to replace them."
        )
    args.features.mkdir(parents=True, exist_ok=True)
    descriptors = extract_elementary_descriptors(args.database)
    successful = descriptors[descriptors["Status"].isin(["OK", "WARNING"])]
    if successful.empty:
        raise ValueError("No meshes produced elementary descriptor values")
    scaler = elementary_scaler(descriptors)
    descriptors.to_csv(outputs[0], index=False)
    scaler.to_csv(outputs[1], index=False)
    config_path = mmru.write_feature_config(args.features, "elementary", {
        "values": "raw",
        "scaling": "z-score",
        "scaler_file": outputs[1].name,
    })
    print(f"Meshes: {len(descriptors)}; retained: {len(successful)}")
    print(f"Meshes with quality warnings: {(descriptors['Status'] == 'WARNING').sum()}")
    print(f"Meshes with extraction errors: {(descriptors['Status'] == 'ERROR').sum()}")
    print(f"Raw descriptors: {outputs[0]}")
    print(f"Z-score parameters: {outputs[1]}")
    print(f"Feature settings: {config_path}")


if __name__ == "__main__":
    main()
