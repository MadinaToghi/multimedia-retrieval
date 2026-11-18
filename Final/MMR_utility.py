"""Small infrastructure helpers shared by the final MMR pipeline modules."""

import hashlib
import json
import random
from pathlib import Path

import trimesh as tr


FEATURE_CONFIG_FILENAME = "feature_config.json"


def show_mesh(mesh, path, isSamewindow=True):
    """Open a mesh in Vedo with shaded faces and a wireframe overlay."""
    from vedo import Mesh, show

    vedo_mesh = Mesh([mesh.vertices, mesh.faces])
    vedo_mesh.c("lightblue")
    vedo_mesh.compute_normals()
    edges = vedo_mesh.clone().wireframe().c("black").lw(1)
    show(
        vedo_mesh,
        edges,
        title=Path(path).name,
        axes=1,
        viewup="z",
        interactive=True,
        new=isSamewindow,
    )


def find_obj_files(database_path):
    """Return sorted OBJ paths below a database root.

    Args:
        database_path: Directory containing category folders and OBJ files.
    Returns:
        Stable, lexically sorted mesh paths.
    """
    database_path = Path(database_path)
    if not database_path.is_dir():
        raise FileNotFoundError(f"Mesh database not found: {database_path}")
    return sorted(database_path.rglob("*.obj"))


def stratified_mesh_sample(mesh_paths, samples_per_category, seed=43):
    """Select a reproducible number of meshes from each category.

    Args:
        mesh_paths: Iterable of OBJ paths.
        samples_per_category: Maximum selection per parent-folder category.
        seed: Random seed for the stratified selection.
    Returns:
        Selected paths, sorted by category and filename.
    """
    if samples_per_category < 1:
        raise ValueError("samples_per_category must be at least one")
    grouped = {}
    for path in sorted(map(Path, mesh_paths)):
        grouped.setdefault(path.parent.name, []).append(path)
    rng = random.Random(seed)
    selected = []
    for category in sorted(grouped):
        paths = grouped[category]
        selected.extend(rng.sample(paths, min(samples_per_category, len(paths))))
    return sorted(selected)


def mesh_identifier(mesh_path, root=None):
    """Create a stable relative identifier for deterministic feature sampling.

    Args:
        mesh_path: Mesh path to identify.
        root: Optional database root to retain relative subfolders.
    Returns:
        POSIX-style mesh identifier, normally category/filename.
    """
    mesh_path = Path(mesh_path).resolve()
    if root is not None:
        try:
            return mesh_path.relative_to(Path(root).resolve()).as_posix()
        except ValueError:
            pass
    return (Path(mesh_path.parent.name) / mesh_path.name).as_posix()


def stable_seed(identifier, base_seed=42):
    """Return a stable integer seed derived from a mesh identifier.

    Args:
        identifier: Stable mesh identifier, normally category/filename.
        base_seed: Seed namespace shared by the pipeline.
    Returns:
        Deterministic non-negative seed.
    """
    digest = hashlib.blake2b(
        f"{base_seed}:{identifier}".encode("utf-8"), digest_size=8
    ).digest()
    return int.from_bytes(digest, "little") % (2**32)


def load_mesh(mesh_path):
    """Load one OBJ as an unprocessed, non-empty triangular Trimesh.

    Args:
        mesh_path: OBJ file to load.
    Returns:
        Loaded Trimesh object.
    """
    mesh = tr.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, tr.Trimesh) or not len(mesh.vertices) or not len(mesh.faces):
        raise ValueError(f"Mesh is empty or invalid: {mesh_path}")
    return mesh


def write_feature_config(features_path, section, values):
    """Record one feature extractor's settings beside its CSV outputs.

    Args:
        features_path: Directory containing feature outputs.
        section: Feature group name, such as histogram or curvature.
        values: JSON-serializable extractor settings.
    Returns:
        Path to the updated feature_config.json file.
    """
    features_path = Path(features_path)
    features_path.mkdir(parents=True, exist_ok=True)
    config_path = features_path / FEATURE_CONFIG_FILENAME
    config = {}
    if config_path.is_file():
        with config_path.open(encoding="utf-8") as config_file:
            config = json.load(config_file)
    config[section] = values
    with config_path.open("w", encoding="utf-8") as config_file:
        json.dump(config, config_file, indent=2, sort_keys=True)
        config_file.write("\n")
    return config_path


def load_feature_config(features_path):
    """Load settings recorded by the feature extractors.

    Args:
        features_path: Directory containing feature outputs.
    Returns:
        Config mapping with histogram and curvature extraction settings.
    Raises:
        FileNotFoundError: If feature extraction has not written a config.
        ValueError: If required extraction settings are missing.
    """
    config_path = Path(features_path) / FEATURE_CONFIG_FILENAME
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Feature settings not found: {config_path}; run the extractors first"
        )
    with config_path.open(encoding="utf-8") as config_file:
        config = json.load(config_file)
    missing_sections = {"histogram", "curvature"} - set(config)
    if missing_sections:
        raise ValueError(
            f"Feature settings are missing sections: {sorted(missing_sections)}"
        )
    return config
