import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from recsys_benchmark.adapters.eager import EAGERAdapter
from recsys_benchmark.config.loader import load_experiment_config
from recsys_benchmark.config.readiness import inspect_method_readiness


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


class EAGERAdapterTest(unittest.TestCase):
    def test_collects_full_ranking_json_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            native_dir = root / "native" / "Beauty"
            native_dir.mkdir(parents=True)
            (native_dir / "eval_results.json").write_text(
                json.dumps(
                    {
                        "dataset": "Beauty",
                        "eval_mode": "full",
                        "recall@5": 0.1,
                        "recall@10": 0.2,
                        "ndcg@5": 0.05,
                        "ndcg@10": 0.08,
                    }
                ),
                encoding="utf-8",
            )
            adapter = EAGERAdapter(
                {
                    "method": {
                        "method_id": "eager",
                        "native_metrics": {
                            "type": "eager_eval_json",
                            "log_dir": "{output_dir}/native/{dataset.dataset_id}",
                            "pattern": "eval_results.json",
                        },
                    },
                    "dataset": {"dataset_id": "Beauty", "task": "sdsr"},
                    "evaluation": {"protocol": "full"},
                    "seed": 2024,
                    "output_dir": str(root),
                }
            )

            result = adapter.evaluate()
            record = json.loads((root / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "evaluated")
        self.assertEqual(record["eval_input_type"], "native_metrics")
        self.assertAlmostEqual(record["metrics"]["recall@10"], 0.2)
        self.assertEqual(record["metrics"]["mrr@10"], "N/A")

    def test_beauty_config_renders_all_native_stages(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ, {"RECSYS_DATA_ROOT": str(Path(tmpdir) / "data")}, clear=False
        ):
            config = load_experiment_config(
                CONFIGS / "experiments" / "beauty_eager.yaml", config_root=CONFIGS
            )
            config["dry_run"] = True
            config["output_dir"] = str(Path(tmpdir) / "run")
            adapter = EAGERAdapter(config)

            prepare = adapter.prepare()
            train = adapter.train()
            predict = adapter.predict()
            readiness = inspect_method_readiness(CONFIGS / "methods" / "eager.yaml")

        self.assertIn("run_sdsr_din.py", " ".join(prepare["command"]))
        self.assertIn("run_sdsr_eager.py", " ".join(train["command"]))
        self.assertIn("eval_sdsr_eager.py", " ".join(predict["command"]))
        self.assertIn("full", predict["command"])
        self.assertEqual(readiness["computed_status"], "adapter-ready")
        self.assertEqual(readiness["declared_status"], "adapter-ready")


if __name__ == "__main__":
    unittest.main()
