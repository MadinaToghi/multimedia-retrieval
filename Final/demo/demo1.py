"""Demo 1: repair, then refine or decimate six raw meshes; saves them for demo 2."""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from demo_common import DATABASE_DIR, DECIMATE_MESHES, REFINE_MESHES, REMESHED_DIR, save_stage
import MMR_utility as mmru
from preprocess import MAX_VERTICES, MIN_VERTICES, repair_mesh, resample_mesh


def draw(axis, mesh, limits, title):
    """Draw the mesh with grey triangle edges."""
    axis.add_collection3d(Poly3DCollection(
        mesh.vertices[mesh.faces], facecolor="lightblue",
        edgecolor=(0, 0, 0, 0.35), linewidth=0.15,
    ))
    centre, radius = limits
    axis.set_xlim(centre[0] - radius, centre[0] + radius)
    axis.set_ylim(centre[1] - radius, centre[1] + radius)
    axis.set_zlim(centre[2] - radius, centre[2] + radius)
    axis.set_box_aspect((1, 1, 1), zoom=1.5)
    axis.view_init(elev=22, azim=-35)
    axis.set_axis_off()
    axis.set_title(title, fontsize=10)


def run_demo(heading, names):
    """Repair and resample each mesh, save the result, and return a before/after figure."""
    figure, axes = plt.subplots(
        len(names), 2, figsize=(13, 4.6 * len(names)), subplot_kw={"projection": "3d"},
    )
    for row, name in enumerate(names):
        raw = mmru.load_mesh(DATABASE_DIR / name)
        repaired, _, _ = repair_mesh(raw)
        remeshed, method, iterations, in_range = resample_mesh(repaired)
        save_stage(remeshed, REMESHED_DIR, name)
        print(
            f"{name:<24} vertices raw {len(raw.vertices):>7,} -> "
            f"repaired {len(repaired.vertices):>7,} -> resampled {len(remeshed.vertices):>7,} | "
            f"{method}, {iterations} iterations | in range: {in_range}"
        )
        limits = (raw.bounds.mean(axis=0), float(np.ptp(raw.bounds, axis=0).max()) / 2)
        for column, (label, shown) in enumerate((("Raw", raw), ("Repaired and resampled", remeshed))):
            draw(axes[row, column], shown, limits, (
                f"{name} - {label}\n{len(shown.vertices):,} vertices"
            ))
    figure.suptitle(f"{heading} (target {MIN_VERTICES:,}-{MAX_VERTICES:,} vertices)")
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    figure.subplots_adjust(hspace=0, wspace=0)


if __name__ == "__main__":
    run_demo("Too few vertices: repair, then subdivide", REFINE_MESHES)
    run_demo("Too many vertices: repair, then simplify", DECIMATE_MESHES)
    print(f"\nSaved to {REMESHED_DIR}; run demo2.py next.")
    plt.show()
