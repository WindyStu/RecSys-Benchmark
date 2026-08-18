import tempfile
import textwrap
import unittest
from pathlib import Path

from recsys_benchmark.config.loader import load_experiment_config


class ConfigLoaderTest(unittest.TestCase):
    def test_merges_dataset_method_experiment_and_cli_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "datasets").mkdir()
            (root / "methods").mkdir()
            (root / "experiments").mkdir()
            (root / "datasets" / "toy.yaml").write_text(
                textwrap.dedent(
                    """
                    dataset_id: toy
                    task: sdsr
                    data_root: examples/toy_sdsr
                    """
                ),
                encoding="utf-8",
            )
            (root / "methods" / "sasrec.yaml").write_text(
                textwrap.dedent(
                    """
                    method_id: sasrec
                    method_type: ranker
                    adapter: recsys_benchmark.adapters.command.CommandAdapter
                    defaults:
                      batch_size: 128
                    """
                ),
                encoding="utf-8",
            )
            experiment = root / "experiments" / "toy_sasrec.yaml"
            experiment.write_text(
                textwrap.dedent(
                    """
                    experiment_id: toy_sasrec
                    dataset: toy
                    method: sasrec
                    seed: 3407
                    evaluation:
                      protocol: full
                    """
                ),
                encoding="utf-8",
            )

            config = load_experiment_config(
                experiment,
                config_root=root,
                overrides={"seed": 1, "evaluation.protocol": "sampled", "method.defaults.batch_size": 64},
            )

        self.assertEqual(config["experiment_id"], "toy_sasrec")
        self.assertEqual(config["dataset"]["dataset_id"], "toy")
        self.assertEqual(config["method"]["method_id"], "sasrec")
        self.assertEqual(config["seed"], 1)
        self.assertEqual(config["evaluation"]["protocol"], "sampled")
        self.assertEqual(config["method"]["defaults"]["batch_size"], 64)
        self.assertEqual(config["dataset"]["path"], str((root / "examples" / "toy_sdsr").resolve()))
        self.assertEqual(config["dataset"]["path_root"], str((root / "examples").resolve()))


if __name__ == "__main__":
    unittest.main()
