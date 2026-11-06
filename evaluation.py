import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("knn_results.csv")
features_df = pd.read_csv("normalized_combined_descriptors.csv")

print("Number of queries:", df["Query_File"].nunique())
print("Neighbors per query:", df.groupby("Query_File").size().value_counts())
print("Self-matches:", (df["Query_File"] == df["File"]).sum())

df["correct"] = df["Query_Class"] == df["Class"]

# Precision for each shape
precision = df.groupby(
    ["Query_File","Query_Class"])["correct"].mean()

precision_df = precision.reset_index(name="precision")

# Overall precision
print("\nOverall precision: ", precision_df["precision"].mean())

# Average precision per class
class_precision = precision_df.groupby(
    "Query_Class"
)["precision"].mean()

print("\nPrecision per class:")
print(class_precision)

# Recall

class_counts = features_df["Class"].value_counts()

# Count correctly retrieved neighbors per query
correct_counts = df.groupby(
    ["Query_File", "Query_Class"])["correct"].sum()

# Convert to a DataFrame
recall_df = correct_counts.reset_index(name="correct_count")

# Number of relevant shapes, excluding the query itself
recall_df["total_relevant"] = recall_df["Query_Class"].map(class_counts) - 1

# Calculate Recall@6
recall_df["recall"] = (
    recall_df["correct_count"] / recall_df["total_relevant"].replace(0, np.nan))

print("\nOverall Recall: ", recall_df["recall"].mean())

class_recall = recall_df.groupby(
    "Query_Class"
)["recall"].mean()

print("\nRecall per class:")
print(class_recall)

# Combine precision and recall for each query
evaluation_df = precision_df.merge(
    recall_df[
        ["Query_File", "Query_Class", "recall"]
    ],
    on=["Query_File", "Query_Class"]
)

# F1 score
p = evaluation_df["precision"] 
r = evaluation_df["recall"] 

evaluation_df["f1"] = 2 * p * r / (p + r).replace(0, np.nan)

# Set f1 to 0 when both are 0 
evaluation_df.loc[
    (p==0) & (r==0), "f1"
] = 0

# Overall F1
print("\nOverall F1: ", evaluation_df["f1"].mean())

# F1 per class
class_f1 = evaluation_df.groupby("Query_Class")["f1"].mean()
print("\nF1 per class: ", class_f1)

# Combine average precision and recall per class
class_evaluation = pd.DataFrame({
    "Precision": class_precision,
    "Recall": class_recall,
    "F1": class_f1
})

evaluation_df.to_csv(
    "step6_evaluation_results.csv",
    index = False
)

# Bar plot
class_precision = class_precision.sort_values(ascending=False)

# Adjust to classes you want
selected_classes = ["Bottle", "Biplane"]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, class_name in zip(axes, selected_classes):

    # Select queries belonging to this class
    class_df = df[df["Query_Class"] == class_name]

    # Calculate percentage of correct results for each rank
    percentages = (
        class_df.groupby("Rank")["correct"].mean() * 100
    )

    # Include all six ranks
    percentages = percentages.reindex(range(1, 7), fill_value=0)

    ax.bar(
        percentages.index,
        percentages.values,
        color="steelblue"
    )

    ax.set_title(class_name, fontsize=14)
    ax.set_xlabel("Rank")
    ax.set_ylabel("Correct retrievals (%)")
    ax.set_xticks(range(1, 7))
    ax.set_ylim(0, 100)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()

plt.savefig(
    "step6_precision_by_rank.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

# Best and worst classes
print("\nBest performing classes:")
print(class_precision.head(5))

print("\nWorst performing classes:")
print(class_precision.tail(5))

print("\nStep 6 evaluation completed!")