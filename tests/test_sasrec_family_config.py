import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from recsys_benchmark.adapters.sasrec_bert4rec import SASRecBERT4RecAdapter
from recsys_benchmark.config.loader import load_experiment_config
from recsys_benchmark.config.readiness import inspect_method_readiness


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = PROJECT_ROOT / "configs"


class SASRecFamilyConfigTest(unittest.TestCase):
    def test_real_beauty_configs_render_prepare_and_train_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_root = Path(tmpdir) / "data"
            for method_id in ("sasrec", "bert4rec", "stosa"):
                with self.subTest(method_id=method_id), patch.dict(
                    os.environ, {"RECSYS_DATA_ROOT": str(data_root)}, clear=False
                ):
                    config = load_experiment_config(
                        CONFIG_ROOT / "experiments" / f"beauty_{method_id}.yaml",
                        config_root=CONFIG_ROOT,
                    )
                    config["dry_run"] = True
                    config["output_dir"] = str(Path(tmpdir) / "runs" / method_id)
                    adapter = SASRecBERT4RecAdapter(config)

                    prepare = adapter.prepare()
                    train = adapter.train()

                    self.assertEqual(config["dataset"]["native_ids"]["sasrec_family"], "abeauty")
                    self.assertIn("single", prepare["command"])
                    self.assertIn(str(data_root / "data_SDSR"), prepare["command"])
                    self.assertIn("--raw", train["command"])
                    self.assertEqual(train["command"][train["command"].index("--m") + 1], method_id)
                    self.assertEqual(train["command"][train["command"].index("--data") + 1], "abeauty")

    def test_all_three_method_configs_are_adapter_ready(self):
        for method_id in ("sasrec", "bert4rec", "stosa"):
            with self.subTest(method_id=method_id):
                report = inspect_method_readiness(CONFIG_ROOT / "methods" / f"{method_id}.yaml")

            self.assertEqual(report["declared_status"], "adapter-ready")
            self.assertEqual(report["computed_status"], "adapter-ready")
            self.assertEqual(report["missing"], [])


if __name__ == "__main__":
    unittest.main()
