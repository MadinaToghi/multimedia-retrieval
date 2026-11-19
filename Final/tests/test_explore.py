import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from explore import find_representative_shapes, read_obj_face_info


class ExploreTests(unittest.TestCase):
    def test_face_info_counts_all_faces_and_detects_mixed_meshes(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "mixed.obj"
            path.write_text(
                "v 0 0 0\n"
                "f 1 2 3\n"
                "f 1 2 3 4\n"
                "f 1 2 3\n",
                encoding="utf-8",
            )
            self.assertEqual(read_obj_face_info(path), (3, "mixed"))

    def test_representative_shapes_identify_count_extremes(self):
        statistics = pd.DataFrame([
            {
                "Class": "A",
                "File": "small.obj",
                "Vertices Count": 10,
                "Faces Count": 8,
            },
            {
                "Class": "B",
                "File": "large.obj",
                "Vertices Count": 100,
                "Faces Count": 80,
            },
            {
                "Class": "C",
                "File": "average.obj",
                "Vertices Count": 55,
                "Faces Count": 44,
            },
        ])
        representatives = find_representative_shapes(statistics)
        selected = dict(zip(
            representatives["Representative"],
            representatives["File"],
        ))
        self.assertEqual(selected["Fewest vertices"], "small.obj")
        self.assertEqual(selected["Most vertices"], "large.obj")
        self.assertEqual(selected["Average-like shape"], "average.obj")


if __name__ == "__main__":
    unittest.main()
