import json
import tempfile
import unittest
from pathlib import Path

from recsys_benchmark.cli import main


class CliTest(unittest.TestCase):
    def test_evaluate_command_writes_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            predictions = root / "topk.csv"
            ground_truth = root / "ground_truth.csv"
            output = root / "metrics.json"
            predictions.write_text("user_id,rank,item_id\nu1,1,i1\n", encoding="utf-8")
            ground_truth.write_text("user_id,item_id\nu1,i1\n", encoding="utf-8")

            exit_code = main(
                [
                    "evaluate",
                    "--predictions",
                    str(predictions),
                    "--ground-truth",
                    str(ground_truth),
                    "--output",
                    str(output),
                    "--input-type",
                    "topk",
                    "--cutoffs",
                    "1",
                    "--method-id",
                    "toy",
                    "--dataset",
                    "toy",
                    "--task",
                    "sdsr",
                    "--protocol",
                    "full",
                    "--seed",
                    "1",
                ]
            )

            record = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertAlmostEqual(record["metrics"]["recall@1"], 1.0)

    def test_inspect_methods_command_writes_readiness_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            methods = root / "methods"
            methods.mkdir()
            source = root / "source"
            source.mkdir()
            (methods / "toy.yaml").write_text(
                f"""
method_id: toy
method_type: ranker
source: {source.as_posix()}
adapter: recsys_benchmark.adapters.command.CommandAdapter
commands:
  train: [python, train.py]
prediction:
  input_type: topk
  path: outputs/topk.csv
""",
                encoding="utf-8",
            )
            output = root / "readiness.json"

            exit_code = main(["inspect-methods", "--methods", str(methods), "--output", str(output)])
            report = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(exit_code, 0)
        self.assertEqual(report[0]["method_id"], "toy")
        self.assertEqual(report[0]["computed_status"], "adapter-ready")


if __name__ == "__main__":
    unittest.main()
