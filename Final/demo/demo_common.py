"""Shared settings for the chained demos: demo1 -> demo2 -> demo3 and demo4."""

import sys
from pathlib import Path

DEMO_DIR = Path(__file__).resolve().parent
PROJECT_DIR = DEMO_DIR.parent
# Allow the scripts to use the pipeline modules when run directly.
sys.path.insert(0, str(PROJECT_DIR))

import MMR_utility as mmru

# Paths relative to the source database root; the first group is also used in
# demo 3, and the second in demo 4.
REFINE_MESHES = ["Drum/D00115.obj", "Guitar/D00097.obj", "Bird/D00089.obj"]
DECIMATE_MESHES = ["Sword/m717.obj", "Computer/m1772.obj", "Mug/D00202.obj"]

DATABASE_DIR = PROJECT_DIR / "shapes" / "ShapeDatabase_INFOMR"
if not DATABASE_DIR.is_dir():
    DATABASE_DIR = PROJECT_DIR.parent.parent / "ShapeDatabase_INFOMR"

DEMO_OUTPUT_DIR = DEMO_DIR / "demo_output"
REMESHED_DIR = DEMO_OUTPUT_DIR / "1_remeshed"
NORMALIZED_DIR = DEMO_OUTPUT_DIR / "2_normalized"


def stage_path(folder, name):
    """Location of a mesh's file inside a demo output folder."""
    return folder / name.replace("/", "_")


def save_stage(mesh, folder, name):
    """Write a mesh for the next demo to pick up."""
    path = stage_path(folder, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    mesh.export(path)


def load_stage(folder, name, previous_demo):
    """Load a mesh written by an earlier demo, or say which demo to run first."""
    path = stage_path(folder, name)
    if not path.is_file():
        raise SystemExit(f"{path.name} not found: run {previous_demo} first.")
    return mmru.load_mesh(path)
