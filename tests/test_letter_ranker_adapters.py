import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from recsys_benchmark.adapters.letter import LetterAdapter
from recsys_benchmark.config.loader import load_experiment_config
from recsys_benchmark.config.readiness import inspect_method_readiness


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


class LetterRankerAdapterTest(unittest.TestCase):
    def test_collects_letter_mean_result_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            native = root / "native"
            native.mkdir()
            (native / "results.json").write_text(
                json.dumps(
                    {
                        "mean_results": {
                            "hit@1": 0.04,
                            "hit@5": 0.12,
                            "hit@10": 0.2,
                            "ndcg@5": 0.07,
                            "ndcg@10": 0.1,
                        }
                    }
                ),
                encoding="utf-8",
            )
            adapter = LetterAdapter(
                {
                    "method": {
                        "method_id": "letter_tiger",
                        "native_metrics": {
                            "type": "letter_result_json",
                            "log_dir": "{output_dir}/native",
                            "pattern": "results.json",
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

        self.assertAlmostEqual(record["metrics"]["recall@5"], 0.12)
        self.assertAlmostEqual(record["metrics"]["ndcg@10"], 0.1)
        self.assertEqual(record["metrics"]["mrr@10"], "N/A")

    def test_tiger_and_lc_rec_configs_render_train_predict_and_are_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ, {"RECSYS_DATA_ROOT": str(Path(tmpdir) / "data")}, clear=False
        ):
            for method_id in ("letter_tiger", "letter_lc_rec"):
                with self.subTest(method_id=method_id):
                    config = load_experiment_config(
                        CONFIGS / "experiments" / f"beauty_{method_id}.yaml", config_root=CONFIGS
                    )
                    config["dry_run"] = True
                    config["output_dir"] = str(Path(tmpdir) / "runs" / method_id)
                    adapter = LetterAdapter(config)
                    train = adapter.train()["command"]
                    predict = adapter.predict()["command"]
                    readiness = inspect_method_readiness(CONFIGS / "methods" / f"{method_id}.yaml")

                    self.assertIn("torchrun", train[0])
                    self.assertIn("--data_path", train)
                    self.assertIn("--results_file", predict)
                    self.assertIn("--metrics", predict)
                    self.assertEqual(readiness["computed_status"], "adapter-ready")
                    self.assertEqual(readiness["declared_status"], "adapter-ready")


if __name__ == "__main__":
    unittest.main()
