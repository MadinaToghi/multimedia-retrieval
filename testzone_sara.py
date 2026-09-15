import numpy as np
import trimesh

shapes_db = "D:/University Project/MMR/ShapeDatabase/" #Change this to your own shape database

mesh = trimesh.load_mesh(shapes_db + "Bed/D00031.obj")

mesh.show()

