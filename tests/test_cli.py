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


if __name__ == "__main__":
    unittest.main()
