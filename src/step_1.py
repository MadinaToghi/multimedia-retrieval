import trimesh as tr
import numpy as np
import random
from pathlib import Path

data_path = Path("../ShapeDatabase_INFOMR")

obj_files = list(data_path.rglob("*.obj"))

random.seed(42)
sample_files = random.sample(obj_files, 10)

meshes = []

for path in sample_files:
    mesh = tr.load(path)
    meshes.append(mesh)
    print(path.name)

meshes[3].show()