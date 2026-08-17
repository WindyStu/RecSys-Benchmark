import tempfile
import unittest
from pathlib import Path

from scripts.extract_metrics import extract_file


class ExtractMetricsTest(unittest.TestCase):
    def test_extracts_single_target_final_metrics_to_rows(self):
        log_text = """
[ Info ] bert4rec-st (1.0 min)
      |                A/B                |
      | hr5    | hr10   | ndcg5  | ndcg10 |  mrr   |
|  F  | 0.1000 | 0.2000 | 0.0700 | 0.1100 | 0.0900 |
"""
        with tempfile.TemporaryDirectory() as tmp:
            log_path = Path(tmp) / "bert4rec-st-abeauty.log"
            log_path.write_text(log_text, encoding="utf-8")

            rows = extract_file(log_path)

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["setting"], "F")
        self.assertEqual(rows[0]["domain"], "A/B")
        self.assertEqual(rows[0]["hr5"], "0.1000")
        self.assertEqual(rows[0]["hr10"], "0.2000")
        self.assertEqual(rows[0]["ndcg5"], "0.0700")
        self.assertEqual(rows[0]["ndcg10"], "0.1100")


if __name__ == "__main__":
    unittest.main()
