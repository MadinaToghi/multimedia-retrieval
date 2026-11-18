"""Load feature tables and search candidates with the query.py distance policy."""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from scipy.stats import wasserstein_distance

from curvature_descriptors import (
    CURVATURE_BIN_CENTERS,
    CURVATURE_BIN_COUNT,
)
from elementary_descriptors import (
    ELEMENTARY_COLUMNS,
    ELEMENTARY_VALIDITY_COLUMNS,
)
from histogram_descriptors import (
    HISTOGRAM_BIN_COUNT,
    HISTOGRAM_DESCRIPTORS,
)


def distribution_distance(query_histogram, candidate_histogram, centers):
    """Compute Wasserstein distance between two binned 1-D distributions."""
    return float(wasserstein_distance(
        centers,
        centers,
        u_weights=query_histogram,
        v_weights=candidate_histogram,
    ))


def histogram_l1_embedding(histogram, centers):
    """Embed a normalized 1-D histogram so L1 equals its Wasserstein distance."""
    histogram = np.asarray(histogram, dtype=np.float64)
    centers = np.asarray(centers, dtype=np.float64)
    if histogram.ndim != 1 or centers.ndim != 1:
        raise ValueError("Histogram and bin centers must be one-dimensional")
    if len(histogram) != len(centers) or len(histogram) < 2:
        raise ValueError("Histogram and bin centers must have matching bins")
    if not np.all(np.isfinite(histogram)) or not np.all(np.isfinite(centers)):
        raise ValueError("Histogram and bin centers must be finite")
    if np.any(np.diff(centers) <= 0):
        raise ValueError("Histogram bin centers must be strictly increasing")
    total = float(histogram.sum())
    if total <= 0:
        raise ValueError("Histogram must have positive mass")
    proportions = histogram / total
    return np.cumsum(proportions)[:-1] * np.diff(centers)


class FeatureSearchIndex:
    """Cache database features and provide brute-force, KNN, and RNN search."""

    def __init__(self, database_path, features_path, descriptor_weights):
        self.database_path = Path(database_path).resolve()
        self.features_path = Path(features_path).resolve()
        self.descriptor_weights = descriptor_weights
        self.descriptors = (
            list(ELEMENTARY_COLUMNS)
            + list(HISTOGRAM_DESCRIPTORS)
            + ["Curvature"]
        )
        self.scaler = pd.read_csv(
            self.features_path / "elementary_scaler.csv"
        ).set_index("Feature")
        self.centers = self._load_centers()
        self.candidates_before_curvature = self._load_candidates()
        curvature_columns = self._histogram_columns(
            "Curvature", CURVATURE_BIN_COUNT
        )
        curvature = self.candidates_before_curvature[
            curvature_columns
        ].to_numpy(dtype=np.float64)
        usable_curvature = np.isfinite(curvature).all(axis=1)
        usable_curvature &= curvature.sum(axis=1) > 0
        self.usable_curvature_mask = usable_curvature
        self.excluded_missing_curvature = int((~usable_curvature).sum())
        self.candidates = (
            self.candidates_before_curvature.loc[usable_curvature]
            .reset_index(drop=True)
        )
        if self.candidates.empty:
            raise ValueError("No candidates have a usable curvature histogram")

        self.candidate_indices = {
            (row["Class"], row["File"]): index
            for index, row in self.candidates[["Class", "File"]].iterrows()
        }
        self.tree = None
        self._candidate_vectors = None
        self.tree_unavailable_reason = self._build_tree()

    @property
    def feature_vectors(self):
        """Return copies of candidate vectors used by the exact L1 search."""
        if self._candidate_vectors is None:
            raise ValueError(
                "Feature vectors are unavailable because the database "
                "contains incomplete descriptors"
            )
        return self._candidate_vectors.copy()

    @staticmethod
    def _histogram_columns(descriptor, bin_count):
        return [
            f"{descriptor}_bin_{index}" for index in range(bin_count)
        ]

    def _load_centers(self):
        frame = pd.read_csv(self.features_path / "histogram_bins.csv")
        centers = {}
        for descriptor in HISTOGRAM_DESCRIPTORS:
            rows = frame.loc[frame["Descriptor"] == descriptor].sort_values("Bin")
            if len(rows) != HISTOGRAM_BIN_COUNT:
                raise ValueError(
                    f"Expected {HISTOGRAM_BIN_COUNT} bins for {descriptor}"
                )
            centers[descriptor] = rows["Center"].to_numpy(dtype=np.float64)
        centers["Curvature"] = CURVATURE_BIN_CENTERS
        return centers

    def _load_candidates(self):
        keys = ["Class", "File"]
        elementary_columns = (
            list(ELEMENTARY_COLUMNS)
            + list(ELEMENTARY_VALIDITY_COLUMNS.values())
            + ["Quality_Warnings"]
        )
        histogram_columns = [
            column
            for descriptor in HISTOGRAM_DESCRIPTORS
            for column in self._histogram_columns(
                descriptor, HISTOGRAM_BIN_COUNT
            )
        ]
        curvature_columns = (
            self._histogram_columns("Curvature", CURVATURE_BIN_COUNT)
            + [
                "Interior_manifold_area_fraction",
                "Successful_fit_sample_fraction",
            ]
        )
        elementary = pd.read_csv(
            self.features_path / "elementary_descriptors.csv"
        )[keys + elementary_columns]
        histograms = pd.read_csv(
            self.features_path / "histogram_descriptors.csv"
        )[keys + histogram_columns]
        curvatures = pd.read_csv(
            self.features_path / "curvature_descriptors.csv"
        )[keys + curvature_columns]

        preprocessing_log = self.database_path / "preprocessing_log.csv"
        if preprocessing_log.is_file():
            candidates = pd.read_csv(preprocessing_log)[keys].drop_duplicates()
        else:
            candidates = elementary[keys].drop_duplicates()
        return candidates.merge(
            elementary, on=keys, how="left", validate="one_to_one"
        ).merge(
            histograms, on=keys, how="left", validate="one_to_one"
        ).merge(
            curvatures, on=keys, how="left", validate="one_to_one"
        )

    def _histogram(self, row, descriptor):
        bin_count = (
            CURVATURE_BIN_COUNT
            if descriptor == "Curvature"
            else HISTOGRAM_BIN_COUNT
        )
        return np.array([
            row[f"{descriptor}_bin_{index}"]
            for index in range(bin_count)
        ], dtype=np.float64)

    def _candidate_vector(self, row):
        parts = []
        for feature in ELEMENTARY_COLUMNS:
            value = row.get(feature, np.nan)
            if not np.isfinite(value):
                raise ValueError(f"Candidate has no finite {feature} value")
            scaler_row = self.scaler.loc[feature]
            std = float(scaler_row["Std"])
            if not np.isfinite(std) or std == 0:
                raise ValueError(f"Invalid standard deviation for {feature}")
            standardized = (
                float(value) - float(scaler_row["Mean"])
            ) / std
            parts.append(np.array([
                standardized * self.descriptor_weights[feature]
            ]))
        for descriptor in HISTOGRAM_DESCRIPTORS + ["Curvature"]:
            histogram = self._histogram(row, descriptor)
            parts.append(
                histogram_l1_embedding(histogram, self.centers[descriptor])
                * self.descriptor_weights[descriptor]
            )
        return np.concatenate(parts)

    def _build_tree(self):
        vectors = []
        try:
            for _, row in self.candidates.iterrows():
                vectors.append(self._candidate_vector(row))
        except ValueError as error:
            return str(error)
        self._candidate_vectors = np.vstack(vectors)
        self.tree = cKDTree(self._candidate_vectors)
        return ""

    def _query_vector(self, query_scaled, query_histograms):
        parts = []
        for feature in ELEMENTARY_COLUMNS:
            value = query_scaled.get(feature, np.nan)
            if not np.isfinite(value):
                raise ValueError(f"Query has no finite {feature} value")
            parts.append(np.array([
                float(value) * self.descriptor_weights[feature]
            ]))
        for descriptor in HISTOGRAM_DESCRIPTORS + ["Curvature"]:
            if descriptor not in query_histograms:
                raise ValueError(f"Query has no {descriptor} histogram")
            parts.append(
                histogram_l1_embedding(
                    query_histograms[descriptor], self.centers[descriptor]
                ) * self.descriptor_weights[descriptor]
            )
        return np.concatenate(parts)

    def _score_candidate(
        self, row, query_scaled, query_histograms, query_weight_total
    ):
        distances = {name: None for name in self.descriptors}
        feature_values = {name: None for name in self.descriptors}
        unavailable = []
        for feature in ELEMENTARY_COLUMNS:
            candidate_value = row.get(feature, np.nan)
            if np.isfinite(candidate_value):
                feature_values[feature] = float(candidate_value)
            if (
                not np.isfinite(query_scaled.get(feature, np.nan))
                or not np.isfinite(candidate_value)
            ):
                unavailable.append(feature)
                continue
            scaler_row = self.scaler.loc[feature]
            candidate_scaled = (
                float(candidate_value) - float(scaler_row["Mean"])
            ) / float(scaler_row["Std"])
            distances[feature] = abs(
                float(query_scaled[feature]) - candidate_scaled
            )

        for descriptor in HISTOGRAM_DESCRIPTORS:
            histogram = self._histogram(row, descriptor)
            if (
                not np.all(np.isfinite(histogram))
                or histogram.sum() <= 0
                or descriptor not in query_histograms
            ):
                unavailable.append(descriptor)
                continue
            feature_values[descriptor] = histogram.copy()
            distances[descriptor] = distribution_distance(
                query_histograms[descriptor],
                histogram,
                self.centers[descriptor],
            )

        curvature = self._histogram(row, "Curvature")
        if (
            "Curvature" not in query_histograms
            or not np.all(np.isfinite(curvature))
            or curvature.sum() <= 0
        ):
            unavailable.append("Curvature")
        else:
            feature_values["Curvature"] = curvature.copy()
            distances["Curvature"] = distribution_distance(
                query_histograms["Curvature"],
                curvature,
                self.centers["Curvature"],
            )

        available = [
            name for name, distance in distances.items()
            if distance is not None
        ]
        available_weight = sum(
            self.descriptor_weights[name] for name in available
        )
        scale = (
            query_weight_total / available_weight
            if available_weight else 0.0
        )
        weighted = {
            name: (
                self.descriptor_weights[name] * distance * scale
                if distance is not None else None
            )
            for name, distance in distances.items()
        }
        score = (
            sum(weighted[name] for name in available)
            if available else np.inf
        )
        return {
            "Class": row["Class"],
            "File": row["File"],
            "Path": self.database_path / row["Class"] / row["File"],
            "Score": float(score),
            "Distances": distances,
            "Weighted_distances": weighted,
            "Feature_values": feature_values,
            "Unavailable_descriptors": unavailable,
            "Elementary_quality_warnings": row.get("Quality_Warnings", ""),
            "Curvature_coverage": row["Interior_manifold_area_fraction"],
            "Curvature_fit_fraction": row["Successful_fit_sample_fraction"],
        }

    @staticmethod
    def _sort_results(results):
        return sorted(
            results,
            key=lambda result: (
                result["Score"], str(result["Class"]), str(result["File"])
            ),
        )

    def search(
        self,
        query_scaled,
        query_histograms,
        search_method="brute_force",
        top_k=5,
        radius=None,
        query_key=None,
    ):
        """Search candidates and return ranked rows plus search diagnostics."""
        if search_method not in {"brute_force", "knn", "rnn"}:
            raise ValueError(f"Unknown search method: {search_method}")
        if search_method != "rnn" and top_k < 1:
            raise ValueError("Top K must be at least one")
        if search_method == "rnn" and (
            radius is None or not np.isfinite(radius) or radius < 0
        ):
            raise ValueError("RNN radius must be a finite non-negative number")

        excluded_query_candidate = 0
        if query_key is not None:
            identity_match = (
                self.candidates_before_curvature["Class"].eq(query_key[0])
                & self.candidates_before_curvature["File"].eq(query_key[1])
            )
            excluded_query_candidate = int(identity_match.sum())
            excluded_missing_curvature = int(
                np.count_nonzero(
                    identity_match.to_numpy()
                    & ~self.usable_curvature_mask
                )
            )
        else:
            excluded_missing_curvature = 0
        excluded_index = self.candidate_indices.get(query_key)
        eligible_indices = [
            index for index in range(len(self.candidates))
            if index != excluded_index
        ]
        if not eligible_indices:
            raise ValueError("No preprocessed candidates were found")

        query_weight_total = sum(
            self.descriptor_weights[name]
            for name in self.descriptors
            if (
                np.isfinite(query_scaled.get(name, np.nan))
                if name in ELEMENTARY_COLUMNS
                else name in query_histograms
            )
        )
        method_used = search_method
        fallback_reason = ""
        candidate_indices = eligible_indices

        if search_method != "brute_force":
            try:
                if self.tree is None:
                    raise ValueError(self.tree_unavailable_reason)
                query_vector = self._query_vector(
                    query_scaled, query_histograms
                )
            except ValueError as error:
                method_used = "brute_force"
                fallback_reason = str(error)
            else:
                if search_method == "knn":
                    target_count = min(top_k, len(eligible_indices))
                    requested = min(
                        len(self.candidates),
                        target_count + int(excluded_index is not None),
                    )
                    tree_distances, tree_indices = self.tree.query(
                        query_vector, k=requested, p=1, eps=0
                    )
                    tree_distances = np.atleast_1d(tree_distances)
                    tree_indices = np.atleast_1d(tree_indices)
                    valid_distances = [
                        float(distance)
                        for distance, index in zip(tree_distances, tree_indices)
                        if int(index) != excluded_index
                        and int(index) < len(self.candidates)
                    ]
                    if len(valid_distances) < target_count:
                        candidate_indices = eligible_indices
                    else:
                        cutoff = sorted(valid_distances)[target_count - 1]
                        tolerance = 1e-12 * max(1.0, abs(cutoff))
                        candidate_indices = self.tree.query_ball_point(
                            query_vector,
                            r=cutoff + tolerance,
                            p=1,
                            eps=0,
                        )
                        candidate_indices = [
                            int(index) for index in candidate_indices
                            if int(index) != excluded_index
                        ]
                else:
                    tolerance = 1e-12 * max(1.0, abs(float(radius)))
                    candidate_indices = self.tree.query_ball_point(
                        query_vector,
                        r=float(radius) + tolerance,
                        p=1,
                        eps=0,
                    )
                    candidate_indices = [
                        int(index) for index in candidate_indices
                        if int(index) != excluded_index
                    ]

        results = [
            self._score_candidate(
                self.candidates.iloc[index],
                query_scaled,
                query_histograms,
                query_weight_total,
            )
            for index in candidate_indices
        ]
        results = self._sort_results(results)
        if search_method == "rnn":
            results = [
                result for result in results if result["Score"] <= radius
            ]
        elif search_method == "knn":
            results = results[:top_k]
        else:
            results = results[:top_k]
        return results, {
            "Candidates_ranked": len(eligible_indices),
            "Excluded_missing_curvature_candidates":
                self.excluded_missing_curvature
                - excluded_missing_curvature,
            "Excluded_query_candidate": excluded_query_candidate,
            "Search_method_requested": search_method,
            "Search_method_used": method_used,
            "Search_fallback_reason": fallback_reason,
            "RNN_radius": radius if search_method == "rnn" else None,
        }
