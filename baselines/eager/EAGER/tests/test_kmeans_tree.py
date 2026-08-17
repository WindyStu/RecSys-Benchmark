import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from KmeansTree import select_cluster_index


class KmeansTreeTest(unittest.TestCase):
    def test_select_cluster_index_aligns_choices_to_index_device(self):
        index = torch.tensor([10, 11, 12, 13])
        choices = torch.tensor([0, 1, 0, 1])

        selected = select_cluster_index(index, choices, 1)

        self.assertEqual(selected.tolist(), [11, 13])


if __name__ == "__main__":
    unittest.main()
