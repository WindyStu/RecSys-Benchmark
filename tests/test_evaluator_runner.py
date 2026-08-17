import json
import tempfile
import unittest
from pathlib import Path

from recsys_benchmark.evaluator.runner import run_evaluation


class EvaluatorRunnerTest(unittest.TestCase):
    def test_evaluates_prediction_file_and_writes_metrics_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            predictions = root / "topk.csv"
            ground_truth = root / "ground_truth.csv"
            output = root / "metrics.json"
            predictions.write_text(
                "user_id,rank,item_id,score,domain,split\n"
                "u1,1,i1,0.9,A,test\n"
                "u1,2,i2,0.8,B,test\n"
                "u2,1,i3,0.7,B,test\n",
                encoding="utf-8",
            )
            ground_truth.write_text(
                "user_id,item_id\n"
                "u1,i2\n"
                "u2,i3\n",
                encoding="utf-8",
            )

            record = run_evaluation(
                predictions_path=predictions,
                ground_truth_path=ground_truth,
                output_path=output,
                input_type="topk",
                cutoffs=[1, 2],
                metadata={
                    "method_id": "toy_model",
                    "dataset": "toy",
                    "task": "sdsr",
                    "protocol": "full",
                    "seed": 1,
                    "eval_input_type": "topk",
                },
            )

            written = json.loads(output.read_text(encoding="utf-8"))

        self.assertEqual(record["method_id"], "toy_model")
        self.assertAlmostEqual(record["metrics"]["recall@2"], 1.0)
        self.assertEqual(written, record)


if __name__ == "__main__":
    unittest.main()
