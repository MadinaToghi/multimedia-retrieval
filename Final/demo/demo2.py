"""Demo 2: center, align, flip, and scale the demo 1 meshes; saves them for demos 3 and 4."""

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from demo_common import DECIMATE_MESHES, NORMALIZED_DIR, REFINE_MESHES, REMESHED_DIR, load_stage, save_stage
from preprocess import align_to_principal_axes, center_mesh, flip_axes, scale_to_unit

# Same order as normalize_mesh.
STEPS = [
    ("Center", center_mesh),
    ("Align", align_to_principal_axes),
    ("Flip", flip_axes),
    ("Scale", scale_to_unit),
]


def measure(mesh):
    """Numbers the demo prints for one mesh state."""
    vertices = np.asarray(mesh.vertices)
    covariance = np.cov(vertices, rowvar=False)
    off_axis = np.abs(covariance - np.diag(np.diag(covariance))).max()
    heavy_side = (np.sign(vertices) * vertices**2).sum(axis=0)
    return {
        "offset": float(np.linalg.norm(vertices.mean(axis=0))),
        "off_axis": float(off_axis),
        "signs": "".join("+" if value >= 0 else "-" for value in heavy_side),
        "extent": float(mesh.extents.max()),
    }


def describe(step, before, after):
    """One printed line explaining what a step changed."""
    if step == "Center":
        return f"barycenter distance from origin {before['offset']:.4f} -> {after['offset']:.1e}"
    if step == "Align":
        return f"off-axis covariance {before['off_axis']:.2e} -> {after['off_axis']:.2e}"
    if step == "Flip":
        return f"heavy side of x,y,z {before['signs']} -> {after['signs']}"
    return f"largest extent {before['extent']:.3f} -> {after['extent']:.3f}"


def draw(axis, mesh, title):
    """Draw a mesh with its barycenter (black) and x/y/z axes (green/blue/purple)."""
    radius = float(np.abs(mesh.bounds).max()) * 1.05
    axis.add_collection3d(Poly3DCollection(
        mesh.vertices[mesh.faces], facecolor="lightblue",
        edgecolor=(0, 0, 0, 0.15), linewidth=0.2, alpha=0.8,
    ))
    for direction, color in zip(np.eye(3), ("tab:green", "tab:blue", "tab:purple")):
        axis.plot(*zip(np.zeros(3), direction * radius), color=color, linewidth=2)
    axis.scatter(*mesh.vertices.mean(axis=0), color="black", s=40)
    axis.set_xlim(-radius, radius)
    axis.set_ylim(-radius, radius)
    axis.set_zlim(-radius, radius)
    axis.set_box_aspect((1, 1, 1), zoom=1.5)
    axis.view_init(elev=22, azim=-35)
    axis.set_axis_off()
    axis.set_title(title, fontsize=9)


def main():
    # Two figures of three meshes each keep the plots readable.
    for group in (REFINE_MESHES, DECIMATE_MESHES):
        figure, axes = plt.subplots(
            len(group), 2, figsize=(13, 4.6 * len(group)), subplot_kw={"projection": "3d"},
        )
        for row, name in enumerate(group):
            raw = load_stage(REMESHED_DIR, name, "demo1.py")
            mesh = raw
            print(f"\n== {name} ==")
            for step, apply_step in STEPS:
                result = apply_step(mesh)
                print(f"{step:<7}{describe(step, measure(mesh), measure(result))}")
                mesh = result
            save_stage(mesh, NORMALIZED_DIR, name)
            for column, (label, shown) in enumerate((("Before", raw), ("After", mesh))):
                stats = measure(shown)
                draw(axes[row, column], shown, (
                    f"{name} - {label}\nlargest extent {stats['extent']:.2f}, "
                    f"barycenter offset {stats['offset']:.2f}"
                ))
        figure.suptitle("Black dot = barycenter   Green/blue/purple = x/y/z axes")
        figure.tight_layout(rect=(0, 0, 1, 0.97))
        figure.subplots_adjust(hspace=0, wspace=0)

    print(f"\nSaved to {NORMALIZED_DIR}; run demo3.py and demo4.py next.")
    plt.show()


if __name__ == "__main__":
    main()
