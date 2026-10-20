import trimesh as tr
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

# First step: Prepare your data

data_path = Path(
    r"C:\Users\madin\Documents\multimedia-retrieval\ShapeDatabase_INFOMR_normalized_cleaned"
)

obj_files = sorted(data_path.rglob("*.obj"))

# Random number of samples
N = 10000

# Number of Bins
B = 25

    # Step 2: Calculate the five descriptors

np.random.seed(42)

# D2
def D2_descriptor(vertices, N):

    vertex_count = len(vertices)

    if vertex_count < 2:
            raise ValueError("D2 requires at least 2 vertices")
    
    # Select N pairs of random vertex indices
    indices = np.random.randint(0, vertex_count, size=(N, 2))

   # Identify pairs containing repeated vertices

    invalid = (indices[:, 0] == indices[:, 1])    
   
       # Resample invalid triples until all indices are distinct
    while np.any(invalid):
        indices[invalid] = np.random.randint(
            0, vertex_count, size=(invalid.sum(), 2)
        )
   
        invalid = (indices[:, 0] == indices[:, 1]) 

    # Get their coordinates
    points1 = vertices[indices[:, 0]]
    points2 = vertices[indices[:, 1]]

    # Calculate all distances
    distances = np.linalg.norm(points1 - points2, axis=1)
    
    return distances

# D1
def D1_descriptor(vertices, N):

    # Select N random vertex indices
    indices = np.random.randint(0, len(vertices), size=N)

    # Get their coordinates
    points = vertices[indices]

    # Normalized so: calculate distances from the origin (0, 0, 0)
    distances = np.linalg.norm(points, axis=1)
    
    return distances

def D3_descriptor(vertices, N):
    vertex_count = len(vertices)

    if vertex_count < 3:
        raise ValueError("D3 requires at least 3 vertices")
    
    # Select N random vertex indices
    indices = np.random.randint(0, vertex_count, size=(N, 3))

    # Identify triples containing repeated vertices
    invalid = (
        (indices[:, 0] == indices[:, 1]) |
        (indices[:, 0] == indices[:, 2]) |
        (indices[:, 1] == indices[:, 2])
    )

    # Resample invalid triples until all indices are distinct
    while np.any(invalid):
        indices[invalid] = np.random.randint(
            0, vertex_count, size=(invalid.sum(), 3)
        )

        invalid = (
            (indices[:, 0] == indices[:, 1]) |
            (indices[:, 0] == indices[:, 2]) |
            (indices[:, 1] == indices[:, 2])
        )

    # Get coordinates
    points1 = vertices[indices[:, 0]]
    points2 = vertices[indices[:, 1]]
    points3 = vertices[indices[:, 2]]

    # Vectors
    v1 = points2 - points1
    v2 = points3 - points1

    # Cross product to get the area of the triangle formed by the three points
    cross_product = np.cross(v1, v2)
    area = np.linalg.norm(cross_product, axis=1) / 2.0

    # Calculate square root of triangle areas
    D3_values = np.sqrt(area)  # Using the square root of the area as a distance measure

    return D3_values

def D4_descriptor(vertices, N):

    vertex_count = len(vertices)
    
    if vertex_count < 4:
        raise ValueError("D4 requires at least 4 vertices")

    # Select N random vertex indices
    indices = np.random.randint(0, vertex_count, size=(N, 4))

    # Identify quadruples containing repeated vertices
    invalid = (
            (indices[:, 0] == indices[:, 1]) |
            (indices[:, 0] == indices[:, 2]) |
            (indices[:, 0] == indices[:, 3]) |
            (indices[:, 1] == indices[:, 2]) |
            (indices[:, 1] == indices[:, 3]) |
            (indices[:, 2] == indices[:, 3])
        )
    
        # Resample invalid quadruples until all indices are distinct
    while np.any(invalid):
        indices[invalid] = np.random.randint(
            0, vertex_count, size=(invalid.sum(), 4)
        )
    
        invalid = (
            (indices[:, 0] == indices[:, 1]) |
            (indices[:, 0] == indices[:, 2]) |
            (indices[:, 0] == indices[:, 3]) |
            (indices[:, 1] == indices[:, 2]) |
            (indices[:, 1] == indices[:, 3]) |
            (indices[:, 2] == indices[:, 3])
        )
    

    # Get coordinates
    points1 = vertices[indices[:, 0]]
    points2 = vertices[indices[:, 1]]
    points3 = vertices[indices[:, 2]]
    points4 = vertices[indices[:, 3]]

    # Calculate the volume of the tetrahedron formed by the four points
    v1 = points2 - points1
    v2 = points3 - points1
    v3 = points4 - points1

    # Volume formula for tetrahedron: V = |(v1 . (v2 x v3))| / 6
    cross_product = np.cross(v2, v3)
    volume = np.abs(np.einsum('ij,ij->i', v1, cross_product)) / 6.0

    D4_values = np.cbrt(volume)  # Using the volume as a distance measure

    return D4_values

def A3_descriptor(vertices, N):

    vertex_count = len(vertices)

    if vertex_count < 3:
        raise ValueError("A3 requires at least 3 vertices")

    # Select N random vertex indices
    indices = np.random.randint(0, vertex_count, size=(N, 3))

    # Identify triples containing repeated vertices
    invalid = (
        (indices[:, 0] == indices[:, 1]) |
        (indices[:, 0] == indices[:, 2]) |
        (indices[:, 1] == indices[:, 2])
    )
    
        # Resample invalid triples until all indices are distinct
    while np.any(invalid):
        indices[invalid] = np.random.randint(
            0, vertex_count, size=(invalid.sum(), 3)
        )
    
        invalid = (
            (indices[:, 0] == indices[:, 1]) |
            (indices[:, 0] == indices[:, 2]) |
            (indices[:, 1] == indices[:, 2])
        )

    # Reject samples that contain identical coordinates
    while True:
        points1 = vertices[indices[:, 0]]
        points2 = vertices[indices[:, 1]]
        points3 = vertices[indices[:, 2]]

        invalid_geometry = (
            (np.linalg.norm(points2 - points1, axis=1) == 0) |
            (np.linalg.norm(points3 - points1, axis=1) == 0)
     )

        if not np.any(invalid_geometry):
            break

        # Resample only the invalid triples
        count = int(invalid_geometry.sum())

        indices[invalid_geometry] = np.random.randint(
            0, vertex_count, size=(count, 3)
        )

    # Ensure resampled triples contain distinct indices
        invalid_indices = (
            (indices[:, 0] == indices[:, 1]) |
            (indices[:, 0] == indices[:, 2]) |
            (indices[:, 1] == indices[:, 2])
        )

        while np.any(invalid_indices):
            indices[invalid_indices] = np.random.randint(
            0, vertex_count, size=(int(invalid_indices.sum()), 3)
        )

        invalid_indices = (
            (indices[:, 0] == indices[:, 1]) |
            (indices[:, 0] == indices[:, 2]) |
            (indices[:, 1] == indices[:, 2])
        )

    # Calculate vectors using valid samples
    v1 = points2 - points1
    v2 = points3 - points1


    # calculate the angle between vectors using the dot product
    numerator = np.sum(v1 * v2, axis = 1)

    denominator = np.linalg.norm(v1, axis = 1) * np.linalg.norm(v2, axis = 1)

    # Avoid division by zero
    cos_angles = np.divide(
        numerator,
        denominator,
        out=np.ones_like(numerator),
        where=denominator > 0
    )
     # Ensure values are between -1 and 1
    cos_angles = np.clip(cos_angles, -1, 1)

    A3_values = np.arccos(cos_angles)

    return A3_values


all_D1 = {}
all_D2 = {}
all_D3 = {}
all_D4 = {}
all_A3 = {}

for path in obj_files: 
    mesh = tr.load(path, force='mesh', process=False)
    vertices = mesh.vertices

    # Calculate the D1 descriptor
    D1 = D1_descriptor(vertices, N)
    all_D1[str(path)] = D1

     # Calculate the D2 descriptor
    D2 = D2_descriptor(vertices, N)
    all_D2[str(path)] = D2

    # Calculate the D3 descriptor
    D3 = D3_descriptor(vertices, N)
    all_D3[str(path)] = D3

    # Calculate the D4 descriptor
    D4 = D4_descriptor(vertices, N)
    all_D4[str(path)] = D4

    # Calculate the A3 descriptor
    A3 = A3_descriptor(vertices, N)
    all_A3[str(path)] = A3

# Global minimum and maximum D1
D1_min_distance = min(min(distances) for distances in all_D1.values())
D1_max_distance = max(max(distances) for distances in all_D1.values())

# Global minimum and maximum D2
D2_min_distance = min(min(distances) for distances in all_D2.values())
D2_max_distance = max(max(distances) for distances in all_D2.values())

# Global minimum and maximum D3
D3_min = min(min(distances) for distances in all_D3.values())
D3_max = max(max(distances) for distances in all_D3.values())

D4_min = min(min(distances) for distances in all_D4.values())
D4_max = max(max(distances) for distances in all_D4.values())

A3_min = min(min(distances) for distances in all_A3.values())
A3_max = max(max(distances) for distances in all_A3.values())
    
# Use fixed bin edges for all histograms!
D2_bin_edges = np.linspace(D2_min_distance, D2_max_distance, B + 1)  # Create B bins between min and max distances

# Use fixed bin edges for all histograms!
D1_bin_edges = np.linspace(D1_min_distance, D1_max_distance, B + 1)  # Create B bins between min and max distances

# Use fixed bin edges for all histograms!
D3_bin_edges = np.linspace(D3_min, D3_max, B + 1)  # Create B bins between min and max distances

D4_bin_edges = np.linspace(D4_min, D4_max, B + 1)  # Create B bins between min and max distances

A3_bin_edges = np.linspace(0, np.pi, B + 1)  # Create B bins between min and max distances

# For each descriptor: A3, D1, D2, D3, D4, repeat the calculation N times and store all the results in a list

calculations = []

for path, distances in all_D2.items():

    # D2 histogram 
    D2_hist, _ = np.histogram(distances, bins=D2_bin_edges)
    D1_hist, _ = np.histogram(all_D1[path], bins=D1_bin_edges)
    D3_hist, _ = np.histogram(all_D3[path], bins=D3_bin_edges)
    D4_hist, _ = np.histogram(all_D4[path], bins=D4_bin_edges)
    A3_hist, _ = np.histogram(all_A3[path], bins=A3_bin_edges)

    D1_hist = D1_hist/D1_hist.sum()
    D2_hist = D2_hist/D2_hist.sum()
    D3_hist = D3_hist/D3_hist.sum()
    D4_hist = D4_hist/D4_hist.sum()
    A3_hist = A3_hist/A3_hist.sum()



    calculations.append({
        "file_path": path,
        "A3": A3_hist,
        "D1": D1_hist,
        "D2": D2_hist,
        "D3": D3_hist,
        "D4": D4_hist
    })
    
# Validation checks
print(len(calculations))  # Expected: 690

print(len(calculations[0]["D1"])) # Expected: 25
print(len(calculations[0]["D2"])) # Expected: 25
print(len(calculations[0]["D3"])) # Expected: 25
print(len(calculations[0]["D4"])) # Expected: 25
print(len(calculations[0]["A3"])) # Expected: 25

print(sum(calculations[0]["D1"]))  # Expected: approximately 1.0
print(sum(calculations[0]["D2"]))  # Expected: approximately 1.0
print(sum(calculations[0]["D3"]))  # Expected: approximately 1.0
print(sum(calculations[0]["D4"]))  # Expected: approximately 1.0
print(sum(calculations[0]["A3"]))  # Expected: approximately 1.0

print("D1 range:", D1_bin_edges[0], D1_bin_edges[-1])
print("D2 range:", D2_bin_edges[0], D2_bin_edges[-1])
print("D3 range:", D3_bin_edges[0], D3_bin_edges[-1])
print("D4 range:", D4_bin_edges[0], D4_bin_edges[-1])
print("A3 range:", A3_bin_edges[0], A3_bin_edges[-1])

for shape in calculations:
    for descriptor in ["A3", "D1", "D2", "D3", "D4"]:
        hist = shape[descriptor]

        assert len(hist) == B
        assert np.isclose(hist.sum(), 1.0)
        assert np.all(hist >= 0)

print("All normalized histograms passed!")


# Save the results to a CSV file
rows = []

for shape in calculations:
    row = {
    "Class": Path(shape["file_path"]).parent.name,
    "File": Path(shape["file_path"]).name
}

    for descriptor in ["D1", "D2", "D3", "D4", "A3"]:
        for i, value in enumerate(shape[descriptor]):
            row[f"{descriptor}_bin_{i}"] = value

    rows.append(row)

df = pd.DataFrame(rows)
df.to_csv("shape_property_descriptors.csv", index=False)

print(df.shape)  # Expected: (690, 127)
print(df.isnull().sum().sum())  # Expected: 0