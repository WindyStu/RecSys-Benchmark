import importlib.util
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


VQREC_DIR = Path(__file__).resolve().parents[1]
COMPAT_PATH = VQREC_DIR / "recbole_compat.py"

spec = importlib.util.spec_from_file_location("recbole_compat", COMPAT_PATH)
recbole_compat = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(recbole_compat)


class RecBoleCompatTest(unittest.TestCase):
    def test_token_seq_split_points_use_per_row_lengths(self):
        series = pd.Series(
            [
                np.array([], dtype=np.int64),
                np.array([1], dtype=np.int64),
                np.array([2, 3], dtype=np.int64),
            ]
        )

        split_points = recbole_compat.token_seq_split_points(series)

        np.testing.assert_array_equal(split_points, np.array([0, 1]))

    def test_token_seq_lengths_use_per_row_lengths(self):
        series = pd.Series(
            [
                np.array([1], dtype=np.int64),
                np.array([2, 3], dtype=np.int64),
                np.array([4, 5, 6], dtype=np.int64),
            ]
        )

        lengths = recbole_compat.token_seq_lengths(series)

        np.testing.assert_array_equal(lengths, np.array([1, 2, 3]))

    def test_benchmark_presets_uses_instance_item_length_field(self):
        class FeatureType:
            TOKEN = "token"

        class FeatureSource:
            INTERACTION = "interaction"

        class FakeDataset:
            def __init__(self):
                self.config = {"LIST_SUFFIX": "_list"}
                self.inter_feat = {
                    "item_id": np.array([1, 2]),
                    "item_id_list": pd.Series(
                        [
                            np.array([1], dtype=np.int64),
                            np.array([1, 2], dtype=np.int64),
                        ]
                    ),
                }
                self.item_list_length_field = "item_length"
                self.properties = []

            def set_field_property(self, field, ftype, source, length):
                self.properties.append((field, ftype, source, length))

        dataset = FakeDataset()

        recbole_compat.apply_benchmark_presets(
            dataset,
            feature_type=FeatureType,
            feature_source=FeatureSource,
        )

        self.assertEqual(dataset.item_id_list_field, "item_id_list")
        self.assertEqual(dataset.properties, [("item_length", "token", "interaction", 1)])
        np.testing.assert_array_equal(dataset.inter_feat["item_length"], np.array([1, 2]))


if __name__ == "__main__":
    unittest.main()
