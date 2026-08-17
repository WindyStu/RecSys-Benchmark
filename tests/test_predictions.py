import tempfile
import unittest
from pathlib import Path

from recsys_benchmark.evaluator.predictions import load_prediction_file, validate_predictions


class PredictionFormatTest(unittest.TestCase):
    def test_loads_and_validates_topk_csv(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "topk.csv"
            path.write_text(
                "user_id,rank,item_id,score,domain,split\n"
                "u1,1,i1,0.9,A,test\n"
                "u1,2,i2,0.8,B,test\n",
                encoding="utf-8",
            )

            rows = load_prediction_file(path)
            validate_predictions(rows, input_type="topk")

        self.assertEqual(rows[0]["user_id"], "u1")
        self.assertEqual(rows[0]["rank"], 1)
        self.assertEqual(rows[1]["item_id"], "i2")

    def test_rejects_candidate_scores_without_score(self):
        rows = [{"user_id": "u1", "item_id": "i1"}]

        with self.assertRaisesRegex(ValueError, "score"):
            validate_predictions(rows, input_type="candidate_scores")


if __name__ == "__main__":
    unittest.main()
