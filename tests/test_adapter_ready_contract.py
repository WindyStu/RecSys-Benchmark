import json
import sys
import tempfile
import textwrap
import unittest
from pathlib import Path

from recsys_benchmark.adapters.command import CommandAdapter
from recsys_benchmark.config.readiness import inspect_method_readiness


class AdapterReadyContractTest(unittest.TestCase):
    def test_prepare_materializes_data_bindings_before_running_commands(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            dataset = root / "dataset"
            source.mkdir()
            dataset.mkdir()
            (dataset / "toy.inter.json").write_text("[]", encoding="utf-8")
            (source / "prepare.py").write_text(
                "from pathlib import Path\n"
                "assert Path('data/toy/toy.inter.json').exists()\n"
                "Path('prepared.txt').write_text('ok', encoding='utf-8')\n",
                encoding="utf-8",
            )

            adapter = CommandAdapter(
                {
                    "method": {
                        "method_id": "toy",
                        "source": str(source),
                        "commands": {"prepare": [sys.executable, "prepare.py"]},
                        "data_bindings": [
                            {
                                "from": "{dataset.path}/toy.inter.json",
                                "to": "{method.source}/data/{dataset.dataset_id}/toy.inter.json",
                                "mode": "copy",
                            }
                        ],
                    },
                    "dataset": {"dataset_id": "toy", "path": str(dataset)},
                    "output_dir": str(root / "run"),
                }
            )

            result = adapter.prepare()

            self.assertEqual(result["returncode"], 0)
            self.assertTrue((source / "data" / "toy" / "toy.inter.json").exists())
            self.assertTrue((source / "prepared.txt").exists())

    def test_command_stage_fails_fast_when_subprocess_returns_nonzero(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            source = Path(tmpdir) / "source"
            source.mkdir()
            (source / "fail.py").write_text("raise SystemExit(7)\n", encoding="utf-8")
            adapter = CommandAdapter(
                {
                    "method": {
                        "method_id": "toy",
                        "source": str(source),
                        "commands": {"train": [sys.executable, "fail.py"]},
                    },
                    "dataset": {"dataset_id": "toy"},
                    "output_dir": str(Path(tmpdir) / "run"),
                }
            )

            with self.assertRaisesRegex(RuntimeError, "train failed"):
                adapter.train()

    def test_evaluate_stage_uses_unified_evaluator_when_prediction_contract_exists(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            predictions = root / "topk.csv"
            truth = root / "ground_truth.csv"
            predictions.write_text("user_id,rank,item_id\nu1,1,i1\n", encoding="utf-8")
            truth.write_text("user_id,item_id\nu1,i1\n", encoding="utf-8")
            adapter = CommandAdapter(
                {
                    "method": {"method_id": "toy", "source": str(root)},
                    "dataset": {"dataset_id": "toy", "task": "sdsr"},
                    "seed": 3,
                    "evaluation": {"protocol": "full", "cutoffs": [1]},
                    "prediction": {
                        "input_type": "topk",
                        "path": str(predictions),
                        "ground_truth": str(truth),
                    },
                    "output_dir": str(root / "run"),
                }
            )

            result = adapter.evaluate()
            metrics = json.loads((root / "run" / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "evaluated")
        self.assertAlmostEqual(metrics["metrics"]["recall@1"], 1.0)

    def test_evaluate_stage_uses_method_prediction_contract_as_default(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            predictions = root / "topk.csv"
            truth = root / "ground_truth.csv"
            predictions.write_text("user_id,rank,item_id\nu1,1,i1\n", encoding="utf-8")
            truth.write_text("user_id,item_id\nu1,i1\n", encoding="utf-8")
            adapter = CommandAdapter(
                {
                    "method": {
                        "method_id": "toy",
                        "source": str(root),
                        "prediction": {
                            "input_type": "topk",
                            "path": str(predictions),
                            "ground_truth": str(truth),
                        },
                    },
                    "dataset": {"dataset_id": "toy", "task": "sdsr"},
                    "seed": 3,
                    "evaluation": {"protocol": "full", "cutoffs": [1]},
                    "output_dir": str(root / "run"),
                }
            )

            result = adapter.evaluate()

        self.assertEqual(result["status"], "evaluated")

    def test_readiness_inspection_reports_adapter_ready_only_when_contract_is_complete(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "method"
            source.mkdir()
            config_path = root / "method.yaml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    method_id: toy
                    method_type: ranker
                    status: source-integrated
                    source: {source.as_posix()}
                    adapter: recsys_benchmark.adapters.command.CommandAdapter
                    commands:
                      train: [python, train.py]
                    prediction:
                      input_type: topk
                      path: outputs/topk.csv
                    """
                ),
                encoding="utf-8",
            )

            report = inspect_method_readiness(config_path)

        self.assertEqual(report["computed_status"], "adapter-ready")
        self.assertEqual(report["missing"], [])


if __name__ == "__main__":
    unittest.main()
