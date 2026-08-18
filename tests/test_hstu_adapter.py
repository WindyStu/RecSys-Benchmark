import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from recsys_benchmark.adapters.generative_recommenders import GenerativeRecommendersAdapter
from recsys_benchmark.config.loader import load_experiment_config
from recsys_benchmark.config.readiness import inspect_method_readiness


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


class HSTUAdapterTest(unittest.TestCase):
    def test_collects_metrics_from_best_checkpoint(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            checkpoint_dir = root / "native" / "Beauty"
            checkpoint_dir.mkdir(parents=True)
            torch.save(
                {
                    "epoch": 7,
                    "metrics": {"hr@5": 0.1, "hr@10": 0.2, "ndcg@5": 0.05, "ndcg@10": 0.08},
                },
                checkpoint_dir / "best_model.pt",
            )
            adapter = GenerativeRecommendersAdapter(
                {
                    "method": {
                        "method_id": "hstu",
                        "native_metrics": {
                            "type": "hstu_checkpoint",
                            "log_dir": "{output_dir}/native/{dataset.dataset_id}",
                            "pattern": "best_model.pt",
                        },
                    },
                    "dataset": {"dataset_id": "Beauty", "task": "sdsr"},
                    "evaluation": {"protocol": "full"},
                    "seed": 42,
                    "output_dir": str(root),
                }
            )

            adapter.evaluate()
            record = json.loads((root / "metrics.json").read_text(encoding="utf-8"))

        self.assertAlmostEqual(record["metrics"]["recall@10"], 0.2)
        self.assertAlmostEqual(record["metrics"]["ndcg@5"], 0.05)
        self.assertEqual(record["metrics"]["mrr@10"], "N/A")

    def test_beauty_config_uses_sdsr_entrypoint_and_is_adapter_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ, {"RECSYS_DATA_ROOT": str(Path(tmpdir) / "data")}, clear=False
        ):
            config = load_experiment_config(
                CONFIGS / "experiments" / "beauty_hstu.yaml", config_root=CONFIGS
            )
            config["dry_run"] = True
            config["output_dir"] = str(Path(tmpdir) / "run")
            command = GenerativeRecommendersAdapter(config).train()["command"]
            readiness = inspect_method_readiness(CONFIGS / "methods" / "hstu.yaml")

        self.assertIn("run_hstu.py", command)
        self.assertIn("--data_dir", command)
        self.assertIn("--output_dir", command)
        self.assertEqual(readiness["computed_status"], "adapter-ready")
        self.assertEqual(readiness["declared_status"], "adapter-ready")


if __name__ == "__main__":
    unittest.main()
