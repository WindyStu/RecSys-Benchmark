import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from recsys_benchmark.adapters.gencdr import GenCDRAdapter
from recsys_benchmark.config.loader import load_experiment_config
from recsys_benchmark.config.readiness import inspect_method_readiness


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


class GenCDRAdapterTest(unittest.TestCase):
    def test_collects_native_generation_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            native = root / "native"
            native.mkdir()
            (native / "results.json").write_text(
                json.dumps(
                    {
                        "mean_results": {
                            "hit@1": 0.03,
                            "hit@5": 0.14,
                            "hit@10": 0.22,
                            "ndcg@5": 0.08,
                            "ndcg@10": 0.11,
                        }
                    }
                ),
                encoding="utf-8",
            )
            adapter = GenCDRAdapter(
                {
                    "method": {
                        "method_id": "gencdr",
                        "native_metrics": {
                            "type": "letter_result_json",
                            "log_dir": "{output_dir}/native",
                            "pattern": "results.json",
                        },
                    },
                    "dataset": {"dataset_id": "asc", "task": "cdsr"},
                    "evaluation": {"protocol": "full"},
                    "seed": 42,
                    "output_dir": str(root),
                }
            )

            adapter.evaluate()
            record = json.loads((root / "metrics.json").read_text(encoding="utf-8"))

        self.assertAlmostEqual(record["metrics"]["recall@5"], 0.14)
        self.assertAlmostEqual(record["metrics"]["ndcg@10"], 0.11)
        self.assertEqual(record["metrics"]["mrr@10"], "N/A")

    def test_asc_config_renders_full_pipeline_and_is_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ, {"RECSYS_DATA_ROOT": str(Path(tmpdir) / "data")}, clear=False
        ):
            config = load_experiment_config(
                CONFIGS / "experiments" / "asc_gencdr.yaml", config_root=CONFIGS
            )
            config["dry_run"] = True
            config["output_dir"] = str(Path(tmpdir) / "runs" / "gencdr")
            adapter = GenCDRAdapter(config)

            prepare = adapter.prepare()["command"]
            train = adapter.train()["command"]
            predict = adapter.predict()["command"]
            readiness = inspect_method_readiness(CONFIGS / "methods" / "gencdr.yaml")

        self.assertIn("export_index.py", prepare)
        self.assertIn("--data_path", prepare)
        self.assertIn("stage2_letter/finetune.py", train)
        self.assertEqual(train[train.index("--stage") + 1], "pretrain")
        self.assertEqual(train[train.index("--datasets") + 1], "asc")
        self.assertIn("stage2_letter/test.py", predict)
        self.assertIn("--results_file", predict)
        self.assertEqual(readiness["computed_status"], "adapter-ready")
        self.assertEqual(readiness["declared_status"], "adapter-ready")


if __name__ == "__main__":
    unittest.main()
