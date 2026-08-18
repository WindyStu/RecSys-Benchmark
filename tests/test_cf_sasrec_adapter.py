import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from recsys_benchmark.adapters.letter import LetterAdapter
from recsys_benchmark.config.loader import load_experiment_config
from recsys_benchmark.config.readiness import inspect_method_readiness


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SOURCE = ROOT / "baselines" / "letter" / "CF-SASRec"


class CFSASRecAdapterTest(unittest.TestCase):
    def test_dry_run_plans_missing_data_binding_without_materializing_it(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            adapter = LetterAdapter(
                {
                    "method": {
                        "source": str(root),
                        "data_bindings": [
                            {"from": str(root / "missing"), "to": str(root / "target"), "mode": "symlink"}
                        ],
                    },
                    "dry_run": True,
                    "output_dir": str(root / "run"),
                }
            )

            result = adapter.prepare()

        self.assertEqual(result["data_bindings"][0]["status"], "planned")

    def test_native_entrypoint_exposes_seed_argument(self):
        result = subprocess.run(
            [sys.executable, "main.py", "--help"], cwd=SOURCE, text=True, capture_output=True, check=False
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--seed", result.stdout)

    def test_collects_best_full_ranking_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            native = root / "native"
            native.mkdir()
            (native / "cf_sasrec_metrics.best.json").write_text(
                json.dumps(
                    {
                        "epoch": 10,
                        "valid": {"HR@5": 0.09, "HR@10": 0.18, "NDCG@5": 0.04, "NDCG@10": 0.07},
                        "test": {"HR@5": 0.1, "HR@10": 0.2, "NDCG@5": 0.05, "NDCG@10": 0.08},
                    }
                ),
                encoding="utf-8",
            )
            adapter = LetterAdapter(
                {
                    "method": {
                        "method_id": "cf_sasrec",
                        "native_metrics": {
                            "type": "cf_sasrec_best_json",
                            "log_dir": "{output_dir}/native",
                            "pattern": "*.best.json",
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

        self.assertAlmostEqual(record["metrics"]["recall@5"], 0.1)
        self.assertAlmostEqual(record["metrics"]["ndcg@10"], 0.08)
        self.assertEqual(record["metrics"]["mrr@10"], "N/A")

    def test_beauty_config_is_adapter_ready_and_renders_required_arguments(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ, {"RECSYS_DATA_ROOT": str(Path(tmpdir) / "data")}, clear=False
        ):
            config = load_experiment_config(
                CONFIGS / "experiments" / "beauty_cf_sasrec.yaml", config_root=CONFIGS
            )
            config["dry_run"] = True
            config["output_dir"] = str(Path(tmpdir) / "run")
            adapter = LetterAdapter(config)
            prepare = adapter.prepare()
            command = adapter.train()["command"]
            readiness = inspect_method_readiness(CONFIGS / "methods" / "cf_sasrec.yaml")

        self.assertEqual(prepare["data_bindings"][0]["status"], "planned")
        self.assertIn("--train_dir", command)
        self.assertIn("--metrics_path", command)
        self.assertIn("--seed", command)
        self.assertEqual(readiness["computed_status"], "adapter-ready")
        self.assertEqual(readiness["declared_status"], "adapter-ready")


if __name__ == "__main__":
    unittest.main()
