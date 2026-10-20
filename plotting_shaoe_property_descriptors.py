
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("shape_property_descriptors.csv")

# Start with A3 (as in the technical tips)
descriptor = "D4"
B = 25

# X-axis represents the 25 histogram bins (0–24)
x = np.arange(B)

columns = [f"{descriptor}_bin_{i}" for i in range(B)]

classes = [
    "Bicycle",
    "Car",
    "Knife",
    "Ship",
    "Hand",
    "PlantWildNonTree",
    "Tool",
    "Drum",
    "Wheelchair",
    "Biplane",
    "Bird"
]

# Only keep classes that actually exist in your dataset
classes = [c for c in classes if c in df["Class"].unique()]

# Arrange plots in a grid
ncols = 2
nrows = int(np.ceil(len(classes) / ncols))

fig, axes = plt.subplots(
    nrows, ncols,
    figsize=(14, nrows * 3),
    squeeze=False
)

for ax, class_name in zip(axes.flat, classes):
    class_shapes = df[df["Class"] == class_name]

    for _, shape in class_shapes.iterrows():
        y = shape[columns].to_numpy(dtype=float)
        ax.plot(x, y, alpha=0.7, linewidth=1)

    ax.set_title(class_name, fontsize=11, pad=10)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=8)

for ax in axes.flat[len(classes):]:
    ax.axis("off")

fig.suptitle(f"{descriptor} histograms grouped by class",
    fontsize=16,
)

fig.supxlabel("Histogram bin")
fig.supylabel("Normalized frequency")

plt.tight_layout(rect=[0.04, 0.05, 0.98, 0.94], h_pad=4)
plt.show()