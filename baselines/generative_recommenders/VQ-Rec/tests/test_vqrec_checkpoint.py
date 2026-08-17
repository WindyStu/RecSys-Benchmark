import importlib.util
import tempfile
import unittest
from pathlib import Path

import torch


VQREC_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_PATH = VQREC_DIR / "vqrec_checkpoint.py"

spec = importlib.util.spec_from_file_location("vqrec_checkpoint", CHECKPOINT_PATH)
vqrec_checkpoint = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(vqrec_checkpoint)


class LegacyConfig:
    def __init__(self, dataset):
        self.dataset = dataset


class MappingConfig:
    def __init__(self, dataset):
        self._values = {"dataset": dataset}

    def __getitem__(self, key):
        return self._values[key]


class VQRecCheckpointTest(unittest.TestCase):
    def test_load_trusted_checkpoint_reads_legacy_config_objects(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "legacy.pth"
            torch.save(
                {
                    "config": LegacyConfig("Cell_Phones"),
                    "state_dict": {"weight": torch.tensor([1.0])},
                },
                checkpoint_path,
            )

            checkpoint = vqrec_checkpoint.load_trusted_checkpoint(
                str(checkpoint_path), map_location="cpu"
            )

            self.assertEqual(checkpoint["config"].dataset, "Cell_Phones")
            self.assertEqual(checkpoint["state_dict"]["weight"].item(), 1.0)

    def test_trusted_torch_load_context_patches_default_torch_load(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "legacy.pth"
            torch.save({"config": LegacyConfig("Cell_Phones")}, checkpoint_path)

            with vqrec_checkpoint.trusted_torch_load_context():
                checkpoint = torch.load(str(checkpoint_path), map_location="cpu")

            self.assertEqual(checkpoint["config"].dataset, "Cell_Phones")

    def test_checkpoint_dataset_name_handles_dict_config(self):
        checkpoint = {"config": {"dataset": "Cell_Phones"}}

        dataset = vqrec_checkpoint.checkpoint_dataset_name(checkpoint)

        self.assertEqual(dataset, "Cell_Phones")

    def test_checkpoint_dataset_name_handles_recbole_config_like_object(self):
        checkpoint = {"config": MappingConfig("Electronics")}

        dataset = vqrec_checkpoint.checkpoint_dataset_name(checkpoint)

        self.assertEqual(dataset, "Electronics")

    def test_checkpoint_dataset_name_falls_back_when_missing(self):
        checkpoint = {}

        dataset = vqrec_checkpoint.checkpoint_dataset_name(checkpoint)

        self.assertEqual(dataset, "?")


if __name__ == "__main__":
    unittest.main()
