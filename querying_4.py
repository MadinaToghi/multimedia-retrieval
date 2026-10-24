import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import read_data as rd
import shape_property_descriptors as spd
from pathlib import Path


df = pd.read_csv("shape_property_descriptors.csv")

data_path = Path("../ShapeDatabase_INFOMR")
query_path = Path("../ShapeDatabase_INFOMR/Hat/m1629.obj")

mesh = rd.process_single_mesh(query_path, data_path)

rd.show_mesh(mesh, query_path)
