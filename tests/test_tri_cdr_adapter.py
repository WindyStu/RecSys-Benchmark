import importlib
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from recsys_benchmark.adapters.tri_cdr import TriCDRAdapter, parse_tri_cdr_log
from recsys_benchmark.config.loader import load_experiment_config
from recsys_benchmark.config.readiness import inspect_method_readiness


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
SOURCE = ROOT / "baselines" / "tri_cdr"


class TriCDRAdapterTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(SOURCE))

    @classmethod
    def tearDownClass(cls):
        sys.path.remove(str(SOURCE))

    def test_generic_partition_builds_target_leave_one_out_sequences(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "asc.inter.json").write_text(
                json.dumps({"u1": [0, 3, 1, 4, 2, 5, 6], "u2": [0, 3, 4]}),
                encoding="utf-8",
            )
            (data_dir / "map_item.txt").write_text(
                json.dumps(
                    {
                        "s0": [0, 0],
                        "s1": [1, 0],
                        "s2": [2, 0],
                        "c0": [3, 1],
                        "c1": [4, 1],
                        "c2": [5, 1],
                        "c3": [6, 1],
                    }
                ),
                encoding="utf-8",
            )
            spec = importlib.util.spec_from_file_location("tri_cdr_utils", SOURCE / "utils.py")
            utils = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(utils)

            partition = utils.data_partition_generic(data_dir, "asc", target_domain=1)

        train_mix, train_source, train_target, valid, test, mix_pos, source_pos, users, items, interval = partition
        self.assertEqual((users, items, interval), (1, 7, 3))
        self.assertEqual(train_mix[1], [1, 4, 2, 5, 3])
        self.assertEqual(train_source[1], [1, 2, 3])
        self.assertEqual(train_target[1], [4, 5])
        self.assertEqual(valid[1], [6])
        self.assertEqual(test[1], [7])
        self.assertEqual(len(mix_pos[1]), 1)
        self.assertEqual(len(source_pos[1]), 1)

    def test_native_model_supports_configured_length_and_cpu_device(self):
        model_module = importlib.import_module("model")
        args = SimpleNamespace(
            device="cpu",
            hidden_units=4,
            maxlen=4,
            num_blocks=1,
            num_heads=1,
            dropout_rate=0.0,
            temperature=1.0,
            dataset="amazon_game",
        )
        model = model_module.SASRec_V12_time_final(1, 8, args)
        mix = np.array([[1, 2, 5, 6]], dtype=np.int64)
        source = np.array([[0, 0, 1, 2]], dtype=np.int64)
        target = np.array([[0, 0, 5, 6]], dtype=np.int64)
        positions = np.array([[0, 0, 2, 3]], dtype=np.int64)

        outputs = model(
            np.array([1]),
            mix,
            source,
            target,
            np.array([[0, 0, 6, 7]]),
            np.array([[0, 0, 7, 8]]),
            positions,
            positions,
        )
        predictions = model.predict(
            np.array([1]), mix, source, target, np.array([5, 7, 8], dtype=np.int64)
        )

        self.assertEqual(tuple(outputs[3].shape), (1, 4))
        self.assertEqual(tuple(predictions[0].shape), (1, 3))
        self.assertTrue(torch.isfinite(predictions[0]).all())

    def test_parser_selects_best_ndcg_epoch(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "test_performance.txt"
            log_path.write_text(
                "epoch:1, time: 1.0(s), test: NDCG@1: 0.01, NDCG@5: 0.05, NDCG@10: 0.08, "
                "NDCG@20: 0.09, NDCG@50: 0.10, HR@1: 0.02, HR@5: 0.10, HR@10: 0.16, "
                "HR@20: 0.20, HR@50: 0.30, AUC: 0.60, loss: 0.4\n"
                "epoch:2, time: 2.0(s), test: NDCG@1: 0.02, NDCG@5: 0.07, NDCG@10: 0.11, "
                "NDCG@20: 0.12, NDCG@50: 0.14, HR@1: 0.03, HR@5: 0.13, HR@10: 0.21, "
                "HR@20: 0.24, HR@50: 0.34, AUC: 0.64, loss: 0.3\n",
                encoding="utf-8",
            )

            metrics = parse_tri_cdr_log(log_path)

        self.assertAlmostEqual(metrics["recall@5"], 0.13)
        self.assertAlmostEqual(metrics["ndcg@10"], 0.11)
        self.assertAlmostEqual(metrics["auc"], 0.64)
        self.assertEqual(metrics["mrr@10"], "N/A")

    def test_asc_config_renders_sampled_pipeline_and_is_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ, {"RECSYS_DATA_ROOT": str(Path(tmpdir) / "data")}, clear=False
        ):
            config = load_experiment_config(
                CONFIGS / "experiments" / "asc_tri_cdr.yaml", config_root=CONFIGS
            )
            config["dry_run"] = True
            config["output_dir"] = str(Path(tmpdir) / "runs" / "tri_cdr")
            adapter = TriCDRAdapter(config)

            train = adapter.train()["command"]
            readiness = inspect_method_readiness(CONFIGS / "methods" / "tri_cdr.yaml")

        self.assertIn("--data_dir", train)
        self.assertEqual(train[train.index("--target_domain") + 1], "1")
        self.assertIn("--result_path", train)
        self.assertEqual(config["evaluation"]["protocol"], "sampled")
        self.assertEqual(readiness["computed_status"], "adapter-ready")
        self.assertEqual(readiness["declared_status"], "adapter-ready")

    def test_native_cli_help_does_not_require_plotting_dependencies(self):
        completed = subprocess.run(
            [sys.executable, "-B", "Tri_CDR.py", "--help"],
            cwd=SOURCE,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertIn("--data_dir", completed.stdout)


if __name__ == "__main__":
    unittest.main()
