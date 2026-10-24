import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import MMR_utility as MMRU
import trimesh as tr


#df = pd.read_csv("shape_property_descriptors.csv")

database_path = Path("../ShapeDatabase_INFOMR")
query_path = Path("../ShapeDatabase_INFOMR/Hat/m1629.obj")

pre_normal_mesh = tr.load(query_path, force="mesh", process=False)

MMRU.show_mesh(pre_normal_mesh, query_path)

info, mesh = MMRU.process_single_mesh(query_path)

print(info)

if mesh != None:
    MMRU.show_mesh(mesh, query_path)
