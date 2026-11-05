import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pathlib import Path
import MMR_utility as mmru
import trimesh as tr
from scipy.stats import wasserstein_distance
from curvature_histogram import (
    BIN_CENTERS as CURVATURE_BIN_CENTERS,
    HISTOGRAM_COLUMNS as CURVATURE_HISTOGRAM_COLUMNS,
)

data_path = Path("../ShapeDatabase_INFOMR_normalized_cleaned")

# Random number of samples
N = 10000

# Number of bins
B = 25

np.random.seed(42)

df_global_property = pd.read_csv("standardized_global_descriptors.csv")
df_shape_property = pd.read_csv("shape_property_descriptors.csv")

df_shape_property = df_shape_property.drop(columns=["Class"])
df_property = df_global_property.merge(df_shape_property, on="File")
df_property = df_property.rename(columns={"Surface Area": "Surface_Area"})

curvature_data = pd.read_csv("curvature_histograms.csv")
curvature_data = curvature_data.loc[curvature_data["status"] == "OK"]
df_curvature = curvature_data.loc[
    :, ["File", *CURVATURE_HISTOGRAM_COLUMNS]
]
df_property = df_property.merge(
    df_curvature,
    on="File",
    validate="one_to_one",
)

query_path = Path("../ShapeDatabase_INFOMR_normalized_cleaned/AircraftBuoyant/m1347.obj")
info, q_mesh = mmru.process_single_mesh(query_path)

if q_mesh is None:
    raise ValueError(f"Could not process query mesh: {query_path}")

# Global descriptors for the query object.
df_mean_std = pd.read_csv("Mean_Std_Dev_Values.csv")
mean_std = df_mean_std.iloc[0]

q_volume = abs(q_mesh.volume)
if not np.isfinite(q_volume) or np.isclose(q_volume, 0):
    raise ValueError(f"Query mesh has invalid or near-zero volume: {query_path}")

else:
    q_volume = (q_volume - mean_std["Volume Mean"]) / mean_std["Volume Std_Dev"]

q_surface_area = mmru.surface_area(q_mesh)
q_surface_area = (q_surface_area - mean_std["Surface Area Mean"] ) / mean_std["Surface Area Std_Dev"]

q_compactness = mmru.compactness(q_mesh)
q_compactness = (q_compactness - mean_std["Compactness Mean"]) / mean_std["Compactness Std_Dev"]

q_rectangularity = mmru.rectangularity(q_mesh)
q_rectangularity = (q_rectangularity - mean_std["Rectangularity Mean"]) / mean_std["Rectangularity Std_Dev"]

q_diameter = mmru.diameter(q_mesh)
q_diameter = (q_diameter - mean_std["Diameter Mean"]) / mean_std["Diameter Std_Dev"]

q_convexity = mmru.convexity(q_mesh)
q_convexity = (q_convexity - mean_std["Convexity Mean"]) / mean_std["Convexity Std_Dev"]

q_eccentricity = mmru.eccentricity(q_mesh)
q_eccentricity = (q_eccentricity - mean_std["Eccentricity Mean"])/ mean_std["Eccentricity Std_Dev"]

df_min_max = pd.read_csv("min_max_property_descriptors.csv")
min_max = df_min_max.iloc[0]
vertices = q_mesh.vertices

# Calculate query histograms using the same bin ranges as the stored descriptors.
q_D1 = mmru.D1_descriptor(vertices, N)
D1_min = min(min_max["D1_min"], min(q_D1))
D1_max = max(min_max["D1_max"], max(q_D1))
D1_bin_edges = np.linspace(D1_min, D1_max, B + 1)
q_D1_hist, _ = np.histogram(q_D1, bins=D1_bin_edges)
q_D1_hist = q_D1_hist/q_D1_hist.sum()

q_D2 = mmru.D2_descriptor(vertices, N)
D2_min = min(min_max["D2_min"], min(q_D2))
D2_max = max(min_max["D2_max"], max(q_D2))
D2_bin_edges = np.linspace(D2_min, D2_max, B + 1)
q_D2_hist, _ = np.histogram(q_D2, bins=D2_bin_edges)
q_D2_hist = q_D2_hist/q_D2_hist.sum()

q_D3 = mmru.D3_descriptor(vertices, N)
D3_min = min(min_max["D3_min"], min(q_D3))
D3_max = max(min_max["D3_max"], max(q_D3))
D3_bin_edges = np.linspace(D3_min, D3_max, B + 1)
q_D3_hist, _ = np.histogram(q_D3, bins=D3_bin_edges)
q_D3_hist = q_D3_hist/q_D3_hist.sum()

q_D4 = mmru.D4_descriptor(vertices, N)
D4_min = min(min_max["D4_min"], min(q_D4))
D4_max = max(min_max["D4_max"], max(q_D4))
D4_bin_edges = np.linspace(D4_min, D4_max, B + 1)
q_D4_hist, _ = np.histogram(q_D4, bins=D4_bin_edges)
q_D4_hist = q_D4_hist/q_D4_hist.sum()

q_A3 = mmru.A3_descriptor(vertices, N)
A3_bin_edges = np.linspace(0, np.pi, B + 1)
q_A3_hist, _ = np.histogram(q_A3, bins=A3_bin_edges)
q_A3_hist = q_A3_hist/q_A3_hist.sum()

query_curvature = curvature_data.loc[
    (curvature_data["Class"] == query_path.parent.name)
    & (curvature_data["File"] == query_path.name),
    CURVATURE_HISTOGRAM_COLUMNS,
]
if len(query_curvature) != 1:
    raise ValueError(
        f"Expected one successful curvature histogram for {query_path}, "
        f"found {len(query_curvature)}"
    )
q_curvature_hist = query_curvature.iloc[0].to_numpy(dtype=float)

# Descriptor weights used both in ranking and in the comparison table.
wvol  = 0.7
wsuar = 1
wcomp = 1
wrec  = 1
wdia  = 0.8
wcvx  = 0.6
wecc  = 0.7
wd1   = 1
wd2   = 1
wd3   = 1.3
wd4   = 0.5
wa3   = 0.6
wcurv = 1.0

weights = {
    "Volume": wvol,
    "Surface Area": wsuar,
    "Compactness": wcomp,
    "Rectangularity": wrec,
    "Diameter": wdia,
    "Convexity": wcvx,
    "Eccentricity": wecc,
    "D1": wd1,
    "D2": wd2,
    "D3": wd3,
    "D4": wd4,
    "A3": wa3,
    "Curvature": wcurv,
}

query_values = {
    "Volume": q_volume,
    "Surface Area": q_surface_area,
    "Compactness": q_compactness,
    "Rectangularity": q_rectangularity,
    "Diameter": q_diameter,
    "Convexity": q_convexity,
    "Eccentricity": q_eccentricity,
    "D1": "25-bin histogram",
    "D2": "25-bin histogram",
    "D3": "25-bin histogram",
    "D4": "25-bin histogram",
    "A3": "25-bin histogram",
    "Curvature": "25-bin histogram",
}

query_histograms = {
    "D1": q_D1_hist,
    "D2": q_D2_hist,
    "D3": q_D3_hist,
    "D4": q_D4_hist,
    "A3": q_A3_hist,
    "Curvature": q_curvature_hist,
}

distances = []
for row in df_property.itertuples(index=False):
    candidate_values = {
        "Volume": row.Volume,
        "Surface Area": row.Surface_Area,
        "Compactness": row.Compactness,
        "Rectangularity": row.Rectangularity,
        "Diameter": row.Diameter,
        "Convexity": row.Convexity,
        "Eccentricity": row.Eccentricity,
    }

    volume_dis = abs(q_volume - row.Volume)
    surf_area_dis = abs(q_surface_area - row.Surface_Area)
    compact_dis = abs(q_compactness - row.Compactness)
    rect_dis = abs(q_rectangularity - row.Rectangularity)
    diam_dis = abs(q_diameter - row.Diameter)
    convex_dis = abs(q_convexity - row.Convexity)
    ecc_dis = abs(q_eccentricity - row.Eccentricity)

    histogram_distances = {}
    for descriptor in query_histograms:
        candidate_histogram = [
            getattr(row, f"{descriptor}_bin_{index}") for index in range(B)
        ]
        if descriptor == "Curvature":
            histogram_distances[descriptor] = wasserstein_distance(
                CURVATURE_BIN_CENTERS,
                CURVATURE_BIN_CENTERS,
                u_weights=query_histograms[descriptor],
                v_weights=candidate_histogram,
            )
        else:
            histogram_distances[descriptor] = wasserstein_distance(
                query_histograms[descriptor], candidate_histogram
            )

    descriptor_distances = {
        "Volume": volume_dis,
        "Surface Area": surf_area_dis,
        "Compactness": compact_dis,
        "Rectangularity": rect_dis,
        "Diameter": diam_dis,
        "Convexity": convex_dis,
        "Eccentricity": ecc_dis,
        **histogram_distances,
    }

    weighted_distances = {
        descriptor: weights[descriptor] * distance
        for descriptor, distance in descriptor_distances.items()
    }
    distances.append({
        "name": row.File,
        "score": sum(weighted_distances.values()),
        "values": candidate_values,
        "weighted_distances": weighted_distances,
    })

distances.sort(key=lambda result: result["score"])
top_matches = distances[:5]

# Load the query and closest meshes for the single comparison window.
query_category = query_path.parent.name
mesh_results = [{
    "label": f"Query ({query_category}): {query_path.name}",
    "mesh": q_mesh,
}]
for rank, result in enumerate(top_matches, start=1):
    mesh_path = next(data_path.rglob(result["name"]), None)
    if mesh_path is None:
        raise FileNotFoundError(
            f"Could not retrieve matching mesh {result['name']} under {data_path}"
        )

    result["category"] = mesh_path.parent.name
    result["mesh"] = tr.load(mesh_path, force="mesh", process=False)
    mesh_results.append({
        "label": f"#{rank} ({result['category']}): {result['name']}",
        "mesh": result["mesh"],
    })

fig = plt.figure(figsize=(20, 15))
grid = fig.add_gridspec(3, 3, height_ratios=(1, 1, 1.5))
mesh_axes = [
    fig.add_subplot(grid[row, column], projection="3d")
    for row in range(2)
    for column in range(3)
]

for ax, result in zip(mesh_axes, mesh_results):
    mesh = result["mesh"]
    ax.add_collection3d(Poly3DCollection(
        mesh.vertices[mesh.faces],
        facecolor="lightblue",
        edgecolor="black",
        linewidth=0.1,
    ))
    center = mesh.bounds.mean(axis=0)
    radius = max(np.ptp(mesh.bounds, axis=0).max() / 2, 1e-6)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    ax.set_title(result["label"])
    ax.set_axis_off()

table_ax = fig.add_subplot(grid[2, :])
table_ax.axis("off")
table_rows = []
histogram_descriptors = {"D1", "D2", "D3", "D4", "A3", "Curvature"}
for descriptor in weights:
    query_value = query_values[descriptor]
    row_values = [
        descriptor,
        query_value if descriptor in histogram_descriptors else f"{query_value:.6g}",
    ]
    for result in top_matches:
        weighted_distance = result["weighted_distances"][descriptor]
        if descriptor in histogram_descriptors:
            row_values.append(f"weighted distance: {weighted_distance:.6g}")
        else:
            row_values.append(
                f"value: {result['values'][descriptor]:.6g}\n"
                f"weighted distance: {weighted_distance:.6g}"
            )
    table_rows.append(row_values)

table = table_ax.table(
    cellText=table_rows,
    colLabels=["Descriptor", "Query"] + [
        f"#{rank} {result['name']} ({result['category']})"
        for rank, result in enumerate(top_matches, start=1)
    ],
    cellLoc="center",
    loc="center",
    colWidths=[0.13, 0.17] + [0.14] * len(top_matches),
)
for column, result in enumerate(top_matches, start=2):
    if result["category"] == query_category:
        for row in range(len(table_rows) + 1):
            table[(row, column)].set_facecolor("#c6efce")

table.auto_set_font_size(False)
table.set_fontsize(8)
table.scale(1, 2)

fig.suptitle(f"Query and five closest meshes: {query_path.name}")
fig.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()
