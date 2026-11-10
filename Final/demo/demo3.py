"""Demo 3: global descriptors of three demo 2 meshes (screen 1), then their shape histograms (screen 2)."""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from demo_common import NORMALIZED_DIR, REFINE_MESHES, load_stage
import MMR_utility as mmru
from elementary_descriptors import ELEMENTARY_COLUMNS, elementary_descriptors
from histogram_descriptors import (
    histogram_edges_from_samples,
    histogram_from_samples,
    sample_histogram_values,
)


# (mesh name from demo 1, camera elevation, azimuth)
DEMO_MESHES = list(zip(REFINE_MESHES, (20, 80, 80), (-60, -90, -90)))
HISTOGRAM_SAMPLES = 10000  # Random vertex combinations drawn per descriptor.
BIN_COUNT = 25
SEED = 42

COLORS = ["tab:orange", "tab:green", "tab:purple"]
# (value format, what the descriptor measures)
GLOBAL_DESCRIPTORS = {
    "Volume": ("{:.3f}", "space inside the shape"),
    "Surface Area": ("{:.2f}", "size of the outer skin"),
    "Compactness": ("{:.2f}", "1 = perfect sphere, near 0 = spread out"),
    "Rectangularity": ("{:.2f}", "volume / volume of tightest rotated box"),
    "Diameter": ("{:.2f}", "longest distance between two surface points"),
    "Convexity": ("{:.2f}", "volume / convex hull volume, 1 = no dents"),
    "Eccentricity": ("{:.1f}", "longest / shortest spread, 1 = round"),
}
SHAPE_DESCRIPTORS = [
    ("A3", "A3: angle between 3 points"),
    ("D1", "D1: distance to center"),
    ("D2", "D2: distance between 2 points"),
    ("D3", "D3: sqrt of triangle area"),
    ("D4", "D4: cube root of tetra volume"),
]


def draw_mesh(axis, mesh, title, elevation, azimuth):
    """Draw the mesh from the given camera angle."""
    axis.add_collection3d(Poly3DCollection(
        mesh.vertices[mesh.faces], facecolor="lightblue",
        edgecolor=(0, 0, 0, 0.15), linewidth=0.1,
    ))
    radius = float(np.abs(mesh.bounds).max())
    for set_limits in (axis.set_xlim, axis.set_ylim, axis.set_zlim):
        set_limits(-radius, radius)
    axis.set_box_aspect((1, 1, 1), zoom=1.2)
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_axis_off()
    axis.set_title(title, fontsize=11)


meshes = []
for name, elevation, azimuth in DEMO_MESHES:
    mesh = load_stage(NORMALIZED_DIR, name, "demo2.py")
    values = elementary_descriptors(mesh)
    if values["Quality_Warnings"]:
        print(f"{name}: {values['Quality_Warnings']}")
    meshes.append((name, mesh, values, (elevation, azimuth)))
print("Meshes are scaled to a largest size of 1, so all values are relative to that.")

# Screen 1: the global descriptors, one table for all three meshes.
figure = plt.figure(figsize=(14, 9))
grid = figure.add_gridspec(2, 3, height_ratios=(1.1, 1.0))
for column, (name, mesh, values, view) in enumerate(meshes):
    draw_mesh(figure.add_subplot(grid[0, column], projection="3d"), mesh, name, *view)
table_axis = figure.add_subplot(grid[1, :])
table_axis.axis("off")
rows = [
    [descriptor, GLOBAL_DESCRIPTORS[descriptor][1]] + [
        GLOBAL_DESCRIPTORS[descriptor][0].format(values[descriptor])
        for _, _, values, _ in meshes
    ]
    for descriptor in ELEMENTARY_COLUMNS
]
table = table_axis.table(
    cellText=rows,
    colLabels=["Descriptor", "What it measures"] + [name for name, *_ in meshes],
    cellLoc="center", loc="center", colWidths=[0.14, 0.36, 0.167, 0.167, 0.167],
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 2.2)
for column, color in enumerate(COLORS[:len(meshes)], start=2):
    table[(0, column)].set_facecolor(color)
figure.suptitle("Global descriptors", fontsize=14)
figure.tight_layout(rect=(0, 0, 1, 0.96))
plt.show()

# Screen 2: shape histograms; bins are shared so the meshes can be compared.
samples = [
    sample_histogram_values(mesh, HISTOGRAM_SAMPLES, mmru.stable_seed(name, SEED))
    for name, mesh, _, _ in meshes
]
ranges = {
    descriptor: (
        min(float(sample[descriptor].min()) for sample in samples),
        max(float(sample[descriptor].max()) for sample in samples),
    )
    for descriptor, _ in SHAPE_DESCRIPTORS
}
edges = histogram_edges_from_samples(ranges, BIN_COUNT)

figure = plt.figure(figsize=(19, 4.2 * len(meshes)))
grid = figure.add_gridspec(len(meshes), 1 + len(SHAPE_DESCRIPTORS), width_ratios=(1.1, 1, 1, 1, 1, 1))
tallest = {}
histograms = [[histogram_from_samples(sample[descriptor], edges[descriptor])
               for descriptor, _ in SHAPE_DESCRIPTORS] for sample in samples]
for column, (descriptor, _) in enumerate(SHAPE_DESCRIPTORS):
    tallest[descriptor] = max(row[column].max() for row in histograms) * 1.08
for row, (name, mesh, _, view) in enumerate(meshes):
    draw_mesh(figure.add_subplot(grid[row, 0], projection="3d"), mesh, name, *view)
    for column, (descriptor, title) in enumerate(SHAPE_DESCRIPTORS, start=1):
        axis = figure.add_subplot(grid[row, column])
        descriptor_edges = edges[descriptor]
        axis.bar(
            (descriptor_edges[:-1] + descriptor_edges[1:]) / 2,
            histograms[row][column - 1], width=np.diff(descriptor_edges),
            color=COLORS[row % len(COLORS)], edgecolor="white",
        )
        axis.set_ylim(0, tallest[descriptor])
        axis.set_xlim(descriptor_edges[0], descriptor_edges[-1])
        if row == 0:
            axis.set_title(title, fontsize=10)
        if column == 1:
            axis.set_ylabel("share of samples")
figure.suptitle(f"Shape descriptors ({HISTOGRAM_SAMPLES:,} samples, {BIN_COUNT} bins)", fontsize=14)
figure.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()
