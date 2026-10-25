import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import MMR_utility as mmru
import trimesh as tr
import math
from operator import itemgetter

# Random number of samples
N = 10000

# Number of Bins
B = 25

    # Step 2: Calculate the five descriptors

np.random.seed(42)

df_global_property = pd.read_csv("shape_descriptors_cleaned.csv")
df_shape_property = pd.read_csv("shape_property_descriptors.csv")

df_shape_property = df_shape_property.drop(columns=["Class"])
df_property  = df_global_property.merge(df_shape_property, on='File')

df_property = df_property.rename(columns={"Surface Area" : "Surface_Area"})



database_path = Path("../ShapeDatabase_INFOMR_normalized_clean")
query_path = Path("../ShapeDatabase_INFOMR/Hat/m1629.obj")

pre_normal_mesh = tr.load(query_path, force="mesh", process=False)

mmru.show_mesh(pre_normal_mesh, query_path)

info, q_mesh = mmru.process_single_mesh(query_path)


if q_mesh != None:
    mmru.show_mesh(q_mesh, query_path)

# Global descriptors query object
q_volume = abs(q_mesh.volume)

if not np.isfinite(q_volume) or np.isclose(q_volume, 0):
    print("Invalid or near-zero volume")


q_surface_area = mmru.surface_area(q_mesh)


q_compactness = mmru.compactness(q_mesh)

q_rectangularity = mmru.rectangularity(q_mesh)

q_diameter = mmru.diameter(q_mesh)

q_convexity = mmru.convexity(q_mesh)

q_eccentricity = mmru.eccentricity(q_mesh)



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

distances = []
#"""
for row in df_property.itertuples():
    name = row.File

    volume_dis = (np.sqrt(np.square(q_volume - row.Volume)))
    surf_area_dis = (np.sqrt(np.square(q_surface_area - row.Surface_Area)))
    compact_dis = (np.sqrt(np.square(q_compactness - row.Compactness)))
    rect_dis = (np.sqrt(np.square(q_rectangularity - row.Rectangularity)))
    diam_dis = (np.sqrt(np.square(q_diameter - row.Diameter)))
    convex_dis = (np.sqrt(np.square(q_convexity - row.Convexity)))
    ecc_dis = (np.sqrt(np.square(q_eccentricity - row.Eccentricity)))

    obj_dis = volume_dis + surf_area_dis + compact_dis + rect_dis + diam_dis + convex_dis + ecc_dis

    distances.append([name, obj_dis])



distances.sort(key=itemgetter(1))

for object in distances[0:5]:
    object_name = object[0]
    print(object_name)
"""
    mesh_path = next(database_path.rglob(object_name), None)
    print(mesh_path)
    

   

    if mesh_path is not None:
        mesh = tr.load(mesh_path, force="mesh", process=False)

        mmru.show_mesh(mesh, mesh_path)

    else:
        print("cant retrieve object: " + object_name)

#"""
