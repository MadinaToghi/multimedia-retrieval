"""Shared data loading and ranking helpers for retrieval evaluation scripts."""

import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist


EVALUATION_DIR = Path(__file__).resolve().parent
FINAL_DIR = EVALUATION_DIR.parent
PROJECT_DIR = FINAL_DIR.parent
if str(FINAL_DIR) not in sys.path:
    sys.path.insert(0, str(FINAL_DIR))

from query import DESCRIPTOR_WEIGHTS
from scalability import FeatureSearchIndex


def load_ranked_feature_distances(database_path, features_path):
    """Build the current feature index and rank all usable meshes per query."""
    index = FeatureSearchIndex(
        database_path, features_path, DESCRIPTOR_WEIGHTS
    )
    distances = cdist(
        index.feature_vectors,
        index.feature_vectors,
        metric="cityblock",
    )
    np.fill_diagonal(distances, np.inf)

    classes = index.candidates["Class"].to_numpy(dtype=str)
    filenames = index.candidates["File"].to_numpy(dtype=str)
    ranked_indices = [
        np.lexsort((filenames, classes, distances[query_index]))
        for query_index in range(len(classes))
    ]
    return index, distances, ranked_indices


def query_relevance(classes, query_index):
    """Return candidate relevance labels with the query mesh excluded."""
    relevant = classes == classes[query_index]
    relevant = relevant.copy()
    relevant[query_index] = False
    return relevant


def roc_curve_for_query(distances, relevant):
    """Return tie-aware false/true-positive rates and ROC AUC."""
    positive_count = int(relevant.sum())
    negative_count = len(relevant) - positive_count
    if positive_count == 0 or negative_count == 0:
        raise ValueError("A ROC curve requires relevant and irrelevant meshes")

    order = np.argsort(distances, kind="stable")
    sorted_distances = distances[order]
    sorted_relevant = relevant[order].astype(int)
    group_ends = np.r_[
        np.flatnonzero(np.diff(sorted_distances)) + 1, len(order)
    ]
    true_positives = np.cumsum(sorted_relevant)[group_ends - 1]
    false_positives = group_ends - true_positives
    tpr = np.r_[0.0, true_positives / positive_count]
    fpr = np.r_[0.0, false_positives / negative_count]
    auc = float(np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2))
    return fpr, tpr, auc


def precision_recall_for_query(distances, relevant):
    """Return tie-aware recall/precision points and average precision."""
    positive_count = int(relevant.sum())
    if positive_count == 0:
        raise ValueError("A precision-recall curve requires relevant meshes")

    order = np.argsort(distances, kind="stable")
    sorted_distances = distances[order]
    sorted_relevant = relevant[order].astype(int)
    group_ends = np.r_[
        np.flatnonzero(np.diff(sorted_distances)) + 1, len(order)
    ]
    true_positives = np.cumsum(sorted_relevant)[group_ends - 1]
    recall = np.r_[0.0, true_positives / positive_count]
    precision = np.r_[1.0, true_positives / group_ends]
    average_precision = float(np.sum(np.diff(recall) * precision[1:]))
    return recall, precision, average_precision
