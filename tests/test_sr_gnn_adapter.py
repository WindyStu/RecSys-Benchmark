import importlib
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from recsys_benchmark.adapters.sasrec_bert4rec import SASRecBERT4RecAdapter
from recsys_benchmark.config.loader import load_experiment_config
from recsys_benchmark.config.readiness import inspect_method_readiness


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SOURCE = ROOT / "baselines" / "sasrec_bert4rec_st_loo"


class SRGNNAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SOURCE))

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(SOURCE))

    def test_native_model_initializes_and_scores_items_on_cpu(self):
        module = importlib.import_module("models.SR_GNN")
        model = module.SRGNN(
            SimpleNamespace(d_latent=8, n_gnn=1, n_item=6, device=torch.device("cpu"))
        )
        sequences = torch.tensor([[0, 1, 2, 3], [0, 0, 2, 4]], dtype=torch.long)
        lengths = torch.tensor([3, 2], dtype=torch.long)
        targets = torch.tensor([4, 5], dtype=torch.long)

        output = model(sequences, lengths)
        loss = model.calculate_loss(sequences, lengths, targets)
        scores = model.full_sort_predict(sequences, lengths)

        self.assertEqual(tuple(output.shape), (2, 8))
        self.assertEqual(tuple(scores.shape), (2, 7))
        self.assertTrue(torch.isfinite(loss))

    def test_trainer_module_is_available(self):
        module = importlib.import_module("trainers.trainer_SRGNN_st")

        self.assertTrue(hasattr(module, "Trainer"))

    def test_beauty_config_renders_native_pipeline_and_is_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ, {"RECSYS_DATA_ROOT": str(Path(tmpdir) / "data")}, clear=False
        ):
            config = load_experiment_config(
                CONFIGS / "experiments" / "beauty_sr_gnn.yaml", config_root=CONFIGS
            )
            config["dry_run"] = True
            config["output_dir"] = str(Path(tmpdir) / "runs" / "sr_gnn")
            adapter = SASRecBERT4RecAdapter(config)

            prepare = adapter.prepare()["command"]
            train = adapter.train()["command"]
            readiness = inspect_method_readiness(CONFIGS / "methods" / "sr_gnn.yaml")

        self.assertIn("scripts/prepare_sdsr_data.py", prepare)
        self.assertEqual(train[train.index("--m") + 1], "sr_gnn")
        self.assertIn("--eval_mode", train)
        self.assertEqual(readiness["computed_status"], "adapter-ready")
        self.assertEqual(readiness["declared_status"], "adapter-ready")


if __name__ == "__main__":
    unittest.main()
