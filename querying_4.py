import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import MMR_utility as mmru
import trimesh as tr

# Random number of samples
N = 10000

# Number of Bins
B = 25

    # Step 2: Calculate the five descriptors

np.random.seed(42)

#df_global_property = pd.read_csv("shape_descriptors_cleaned.csv")
#df_shape_property = pd.read_csv("shape_property_descriptors.csv")


database_path = Path("../ShapeDatabase_INFOMR")
query_path = Path("../ShapeDatabase_INFOMR/Hat/m1629.obj")

pre_normal_mesh = tr.load(query_path, force="mesh", process=False)

mmru.show_mesh(pre_normal_mesh, query_path)

info, q_mesh = mmru.process_single_mesh(query_path)

print(info)

if q_mesh != None:
    mmru.show_mesh(q_mesh, query_path)

# Global descriptors query object
q_volume = abs(q_mesh.volume)

if not np.isfinite(q_volume) or np.isclose(q_volume, 0):
    print("Invalid or near-zero volume")


q_surface_area = mmru.surface_area(q_mesh)
q_compactness = mmru.compactness(q_mesh),
q_rectangularity = mmru.rectangularity(q_mesh),
q_diameter = mmru.diameter(q_mesh),
q_convexity = mmru.convexity(q_mesh),
q_eccentricity = mmru.eccentricity(q_mesh),

# shape properties query object
vertices = q_mesh.vertices

# Calculate the D1 descriptor
q_D1 = mmru.D1_descriptor(vertices, N)
   
# Calculate the D2 descriptor
q_D2 = mmru.D2_descriptor(vertices, N)
    
# Calculate the D3 descriptor
q_D3 = mmru.D3_descriptor(vertices, N)
    
# Calculate the D4 descriptor
q_D4 = mmru.D4_descriptor(vertices, N)
    
# Calculate the A3 descriptor
q_A3 = mmru.A3_descriptor(vertices, N)