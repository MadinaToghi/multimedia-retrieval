import trimesh as tr
from pathlib import Path
from vedo import Mesh, show

data_path = Path("../ShapeDatabase_INFOMR")

# Selected Step 1 meshes
selected_files = [
    next(data_path.rglob("D00115.obj")),
    next(data_path.rglob("D00870.obj")),
    next(data_path.rglob("m985.obj")),
]

def show_mesh(mesh, path, mode):
    vedo_mesh = Mesh([mesh.vertices, mesh.faces])
    vedo_mesh.c("lightblue")

    if mode == "smooth":
        vedo_mesh.compute_normals()
        actors = [vedo_mesh]

    elif mode == "flat":
        vedo_mesh.flat()
        actors = [vedo_mesh]

    elif mode == "shaded_edges":
        vedo_mesh.compute_normals()

        edges = (
            vedo_mesh.clone()
            .wireframe()
            .c("black")
            .lw(1)
        )

        actors = [vedo_mesh, edges]

    elif mode == "wireframe":
        wire = (
            vedo_mesh
            .wireframe()
            .c("black")
            .lw(1)
        )

        actors = [wire]

    else:
        raise ValueError(f"Unknown mode: {mode}")

    show(
        *actors,
        title=f"{path.name} - {mode}",
        axes=1,
        viewup="z",
        interactive=True,
        new=True
    )


modes = [
    "smooth",
    "flat",
    "shaded_edges",
    "wireframe"
]

for path in selected_files:
    print(f"Displaying: {path.name}")

    mesh = tr.load(
        path,
        force="mesh",
        process=False
    )

    for mode in modes:
        show_mesh(mesh, path, mode)