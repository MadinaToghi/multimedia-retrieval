import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from curvature_descriptors import CURVATURE_BIN_COUNT
from elementary_descriptors import (
    ELEMENTARY_COLUMNS,
    ELEMENTARY_VALIDITY_COLUMNS,
)
from histogram_descriptors import HISTOGRAM_BIN_COUNT, HISTOGRAM_DESCRIPTORS
from scalability import FeatureSearchIndex, histogram_l1_embedding
from query import DESCRIPTOR_WEIGHTS


class RetrievalTests(unittest.TestCase):
    def test_histogram_embedding_matches_wasserstein(self):
        centers = np.array([0.0, 0.4, 1.5, 3.0])
        first = np.array([0.55, 0.1, 0.25, 0.1])
        second = np.array([0.05, 0.25, 0.2, 0.5])
        expected = wasserstein_distance(
            centers, centers, u_weights=first, v_weights=second
        )
        actual = np.abs(
            histogram_l1_embedding(first, centers)
            - histogram_l1_embedding(second, centers)
        ).sum()
        self.assertAlmostEqual(actual, expected, places=14)

    def test_knn_and_rnn_match_brute_force_scores(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            database = root / "database"
            features = root / "features"
            database.mkdir()
            features.mkdir()
            keys = [("Test", f"mesh-{index}.obj") for index in range(3)]
            pd.DataFrame(keys, columns=["Class", "File"]).to_csv(
                database / "preprocessing_log.csv", index=False
            )
            pd.DataFrame({
                "Feature": ELEMENTARY_COLUMNS,
                "Mean": np.zeros(len(ELEMENTARY_COLUMNS)),
                "Std": np.ones(len(ELEMENTARY_COLUMNS)),
            }).to_csv(features / "elementary_scaler.csv", index=False)
            bins = []
            for descriptor in HISTOGRAM_DESCRIPTORS:
                bins.extend({
                    "Descriptor": descriptor,
                    "Bin": index,
                    "Center": index * 0.25,
                } for index in range(HISTOGRAM_BIN_COUNT))
            pd.DataFrame(bins).to_csv(
                features / "histogram_bins.csv", index=False
            )

            elementary_rows = []
            histogram_rows = []
            curvature_rows = []
            base_histogram = np.zeros(HISTOGRAM_BIN_COUNT)
            base_histogram[0] = 0.7
            base_histogram[-1] = 0.3
            for index, (category, filename) in enumerate(keys):
                elementary_rows.append({
                    "Class": category,
                    "File": filename,
                    **{
                        feature: float(index + feature_index / 10)
                        for feature_index, feature in enumerate(
                            ELEMENTARY_COLUMNS
                        )
                    },
                    **{
                        flag: True
                        for flag in ELEMENTARY_VALIDITY_COLUMNS.values()
                    },
                    "Quality_Warnings": "",
                })
                histogram_row = {"Class": category, "File": filename}
                for descriptor in HISTOGRAM_DESCRIPTORS:
                    histogram = np.roll(base_histogram, index)
                    histogram_row.update({
                        f"{descriptor}_bin_{bin_index}": value
                        for bin_index, value in enumerate(histogram)
                    })
                histogram_rows.append(histogram_row)

                curvature = np.zeros(CURVATURE_BIN_COUNT)
                curvature[index] = 0.7
                curvature[-1] += 0.3
                curvature_rows.append({
                    "Class": category,
                    "File": filename,
                    **{
                        f"Curvature_bin_{bin_index}": value
                        for bin_index, value in enumerate(curvature)
                    },
                    "Interior_manifold_area_fraction": 1.0,
                    "Successful_fit_sample_fraction": 1.0,
                })

            pd.DataFrame(elementary_rows).to_csv(
                features / "elementary_descriptors.csv", index=False
            )
            pd.DataFrame(histogram_rows).to_csv(
                features / "histogram_descriptors.csv", index=False
            )
            pd.DataFrame(curvature_rows).to_csv(
                features / "curvature_descriptors.csv", index=False
            )

            index = FeatureSearchIndex(
                database, features, DESCRIPTOR_WEIGHTS
            )
            query_scaled = {
                feature: 0.8 for feature in ELEMENTARY_COLUMNS
            }
            query_histograms = {}
            for descriptor in HISTOGRAM_DESCRIPTORS:
                query_histograms[descriptor] = (
                    base_histogram + np.roll(base_histogram, 1)
                ) / 2
            query_curvature = np.zeros(CURVATURE_BIN_COUNT)
            query_curvature[0] = 0.4
            query_curvature[1] = 0.4
            query_curvature[-1] = 0.2
            query_histograms["Curvature"] = query_curvature

            brute, _ = index.search(
                query_scaled, query_histograms, top_k=3
            )
            knn, knn_info = index.search(
                query_scaled, query_histograms, search_method="knn",
                top_k=2,
            )
            self.assertEqual(knn_info["Search_method_used"], "knn")
            self.assertEqual(
                [(row["Class"], row["File"]) for row in knn],
                [(row["Class"], row["File"]) for row in brute[:2]],
            )
            np.testing.assert_allclose(
                [row["Score"] for row in knn],
                [row["Score"] for row in brute[:2]],
                rtol=1e-12,
                atol=1e-12,
            )

            brute_without_query, brute_info = index.search(
                query_scaled,
                query_histograms,
                top_k=3,
                query_key=keys[0],
            )
            knn_without_query, knn_info = index.search(
                query_scaled,
                query_histograms,
                search_method="knn",
                top_k=3,
                query_key=keys[0],
            )
            self.assertEqual(brute_info["Excluded_query_candidate"], 1)
            self.assertEqual(knn_info["Excluded_query_candidate"], 1)
            self.assertEqual(
                [(row["Class"], row["File"]) for row in knn_without_query],
                [
                    (row["Class"], row["File"])
                    for row in brute_without_query
                ],
            )

            radius = brute[1]["Score"]
            rnn, rnn_info = index.search(
                query_scaled, query_histograms, search_method="rnn",
                radius=radius,
            )
            self.assertEqual(rnn_info["Search_method_used"], "rnn")
            self.assertEqual(
                [(row["Class"], row["File"]) for row in rnn],
                [
                    (row["Class"], row["File"])
                    for row in brute if row["Score"] <= radius
                ],
            )

            missing_curvature = dict(query_histograms)
            missing_curvature.pop("Curvature")
            fallback, fallback_info = index.search(
                query_scaled, missing_curvature, search_method="knn",
                top_k=2,
            )
            self.assertEqual(
                fallback_info["Search_method_used"], "brute_force"
            )
            self.assertTrue(fallback_info["Search_fallback_reason"])
            reference, _ = index.search(
                query_scaled, missing_curvature, top_k=2
            )
            self.assertEqual(
                [(row["Class"], row["File"]) for row in fallback],
                [(row["Class"], row["File"]) for row in reference],
            )


if __name__ == "__main__":
    unittest.main()
