import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import MMR_utility as mmru
import trimesh as tr
import math
from operator import itemgetter
from scipy.stats import wasserstein_distance
import random

from vedo import Mesh, show

data_path = Path("../ShapeDatabase_INFOMR_normalized_cleaned")

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
q_D1_hist, _ = np.histogram(q_D1, bins = B, density=True)

# Calculate the D2 descriptor
q_D2 = mmru.D2_descriptor(vertices, N)
q_D2_hist, _ = np.histogram(q_D2, bins = B, density=True)
    
# Calculate the D3 descriptor
q_D3 = mmru.D3_descriptor(vertices, N)
q_D3_hist, _ = np.histogram(q_D3, bins=B, density=True)

# Calculate the D4 descriptor
q_D4 = mmru.D4_descriptor(vertices, N)
q_D4_hist, _ = np.histogram(q_D4, bins=B, density=True)
    
# Calculate the A3 descriptor
q_A3 = mmru.A3_descriptor(vertices, N)
q_A3_hist, _ = np.histogram(q_A3, bins= B, density=True)

distances = []


wvol = 1
wsuar = 1
wcomp = 1
wrec = 1
wdia = 1
wcvx = 1
wacc = 1

wd1 = 1
wd2 = 1
wd3 = 1
wd4 = 1
wa3 = 1





for row in df_property.itertuples():
    name = row.File

    r_D1 = [row.D1_bin_0, row.D1_bin_1, row.D1_bin_2, row.D1_bin_3, row.D1_bin_4, 
            row.D1_bin_5, row.D1_bin_6, row.D1_bin_7, row.D1_bin_8, row.D1_bin_9,
            row.D1_bin_10,row.D1_bin_11,row.D1_bin_12,row.D1_bin_13,row.D1_bin_14,
            row.D1_bin_15,row.D1_bin_16,row.D1_bin_17,row.D1_bin_18,row.D1_bin_19,
            row.D1_bin_20,row.D1_bin_21,row.D1_bin_22,row.D1_bin_23,row.D1_bin_24]

    r_D2 = [row.D2_bin_0, row.D2_bin_1, row.D2_bin_2, row.D2_bin_3, row.D2_bin_4, 
            row.D2_bin_5, row.D2_bin_6, row.D2_bin_7, row.D2_bin_8, row.D2_bin_9,
            row.D2_bin_10,row.D2_bin_11,row.D2_bin_12,row.D2_bin_13,row.D2_bin_14,
            row.D2_bin_15,row.D2_bin_16,row.D2_bin_17,row.D2_bin_18,row.D2_bin_19,
            row.D2_bin_20,row.D2_bin_21,row.D2_bin_22,row.D2_bin_23,row.D2_bin_24]

    r_D3 = [row.D3_bin_0, row.D3_bin_1, row.D3_bin_2, row.D3_bin_3, row.D3_bin_4, 
            row.D3_bin_5, row.D3_bin_6, row.D3_bin_7, row.D3_bin_8, row.D3_bin_9,
            row.D3_bin_10,row.D3_bin_11,row.D3_bin_12,row.D3_bin_13,row.D3_bin_14,
            row.D3_bin_15,row.D3_bin_16,row.D3_bin_17,row.D3_bin_18,row.D3_bin_19,
            row.D3_bin_20,row.D3_bin_21,row.D3_bin_22,row.D3_bin_23,row.D3_bin_24]
    
    r_D4 = [row.D4_bin_0, row.D4_bin_1, row.D2_bin_2, row.D4_bin_3, row.D4_bin_4, 
            row.D4_bin_5, row.D4_bin_6, row.D2_bin_7, row.D4_bin_8, row.D4_bin_9,
            row.D4_bin_10,row.D4_bin_11,row.D2_bin_12,row.D4_bin_13,row.D4_bin_14,
            row.D4_bin_15,row.D4_bin_16,row.D2_bin_17,row.D4_bin_18,row.D4_bin_19,
            row.D4_bin_20,row.D4_bin_21,row.D2_bin_22,row.D4_bin_23,row.D4_bin_24]
    
    r_A3 = [row.A3_bin_0, row.A3_bin_1, row.A3_bin_2, row.A3_bin_3, row.A3_bin_4, 
            row.A3_bin_5, row.A3_bin_6, row.A3_bin_7, row.A3_bin_8, row.A3_bin_9,
            row.A3_bin_10,row.A3_bin_11,row.A3_bin_12,row.A3_bin_13,row.A3_bin_14,
            row.A3_bin_15,row.A3_bin_16,row.A3_bin_17,row.A3_bin_18,row.A3_bin_19,
            row.A3_bin_20,row.A3_bin_21,row.A3_bin_22,row.A3_bin_23,row.A3_bin_24]

    volume_dis = (np.sqrt(np.square(q_volume - row.Volume)))
    surf_area_dis = (np.sqrt(np.square(q_surface_area - row.Surface_Area)))
    compact_dis = (np.sqrt(np.square(q_compactness - row.Compactness)))
    rect_dis = (np.sqrt(np.square(q_rectangularity - row.Rectangularity)))
    diam_dis = (np.sqrt(np.square(q_diameter - row.Diameter)))
    convex_dis = (np.sqrt(np.square(q_convexity - row.Convexity)))
    ecc_dis = (np.sqrt(np.square(q_eccentricity - row.Eccentricity)))


    # Wasserstein distance = earth mover distance
    D1_dis = wasserstein_distance(q_D1_hist, r_D1)
    D2_dis = wasserstein_distance(q_D2_hist, r_D2)
    D3_dis = wasserstein_distance(q_D3_hist, r_D3)
    D4_dis = wasserstein_distance(q_D4_hist, r_D4)
    A3_dis = wasserstein_distance(q_A3_hist, r_A3)

    obj_dis = volume_dis + surf_area_dis + compact_dis + rect_dis + diam_dis + convex_dis + ecc_dis + D1_dis + D2_dis + D3_dis + D4_dis + A3_dis

    distances.append([name, obj_dis])



distances.sort(key=itemgetter(1))

for object in distances[0:10]:
    object_name = object[0]
    print(object_name)


# Find the file in any class folder
    mesh_path = next(data_path.rglob(object_name), None)
    

    if mesh_path is not None:
        mesh = tr.load(mesh_path, force="mesh", process=False)

        mmru.show_mesh(mesh, mesh_path)

    else:
        print("cant retrieve object: " + object_name)

#"""
