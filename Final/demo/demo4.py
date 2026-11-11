"""Demo 4: curvature histograms of a sharp, a flat, and a round demo 2 mesh, with an adjustable radius."""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from demo_common import DECIMATE_MESHES, NORMALIZED_DIR, load_stage
from curvature_descriptors import (
    CURVATURE_BIN_CENTERS,
    CURVATURE_BIN_EDGES,
    curvature_single,
)

# Change these and rerun to see how they reshape the histograms.
RADIUS = 0.05  # Meshes are scaled to size 1, so 0.05 is 5% of the mesh.
SAMPLE_COUNT = 3000  # Surface points where curvature is estimated.

# (mesh name from demo 1, description)
DEMO_MESHES = list(zip(
    DECIMATE_MESHES, ("sharp edges", "mostly flat", "smooth and round"),
    (80, 18, 18), (-90, -75, -75),
))
SEED = 42

TICK_CURVATURES = [0, 0.5, 1, 2, 5, 10, 30, 100]


def draw_mesh(axis, mesh, title, elevation, azimuth):
    """Draw the mesh from the given camera angle."""
    axis.add_collection3d(Poly3DCollection(
        mesh.vertices[mesh.faces], facecolor="lightblue",
        edgecolor=(0, 0, 0, 0.15), linewidth=0.1,
    ))
    radius = float(np.abs(mesh.bounds).max())
    for set_limits in (axis.set_xlim, axis.set_ylim, axis.set_zlim):
        set_limits(-radius, radius)
    axis.set_box_aspect((1, 1, 1), zoom=1.5)
    axis.view_init(elev=elevation, azim=azimuth)
    axis.set_axis_off()
    axis.set_title(title, fontsize=10)


def draw_histogram(axis, histogram, color):
    """Bar chart of curvature proportions; the x axis is log-spaced like the bins."""
    axis.bar(
        CURVATURE_BIN_CENTERS, histogram,
        width=np.diff(CURVATURE_BIN_EDGES), color=color, edgecolor="white",
    )
    axis.set_xticks(np.log1p(TICK_CURVATURES), [str(value) for value in TICK_CURVATURES])
    axis.set_xlabel("absolute mean curvature (1 / mesh size)")
    axis.set_ylabel("share of surface")


print(f"Radius {RADIUS}, {SAMPLE_COUNT} sample points")
results = []
for name, description, elevation, azimuth in DEMO_MESHES:
    mesh = load_stage(NORMALIZED_DIR, name, "demo2.py")
    histogram, coverage, fit_fraction = curvature_single(
        mesh, seed=SEED, radius=RADIUS, query_sample_count=SAMPLE_COUNT
    )
    typical = float(np.expm1(histogram @ CURVATURE_BIN_CENTERS))
    print(
        f"{name:<24} typical curvature {typical:5.2f} | "
        f"surface used {coverage:.0%} | fits succeeded {fit_fraction:.0%}"
    )
    results.append((name, description, mesh, histogram, typical, (elevation, azimuth)))

figure = plt.figure(figsize=(13, 4.4 * len(results)))
grid = figure.add_gridspec(len(results), 2, width_ratios=(1.2, 1.4))
tallest = max(result[3].max() for result in results) * 1.08
colors = ["tab:red", "tab:gray", "tab:blue"]
for row, (name, description, mesh, histogram, typical, view) in enumerate(results):
    draw_mesh(
        figure.add_subplot(grid[row, 0], projection="3d"), mesh,
        f"{name} ({description})", *view,
    )
    axis = figure.add_subplot(grid[row, 1])
    draw_histogram(axis, histogram, colors[row % len(colors)])
    axis.set_ylim(0, tallest)
    axis.set_title(f"typical curvature about {typical:.2f}", fontsize=10)
figure.suptitle(f"Curvature histogram, neighbourhood radius = {RADIUS}", fontsize=13)
figure.tight_layout(rect=(0, 0, 1, 0.97))
plt.show()
