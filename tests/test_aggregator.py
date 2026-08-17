import json
import tempfile
import unittest
from pathlib import Path

from recsys_benchmark.aggregator.results import aggregate_runs


class AggregatorTest(unittest.TestCase):
    def test_aggregates_matching_runs_and_keeps_protocols_separate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for idx, (run_id, protocol, recall) in enumerate(
                [
                    ("run1", "full", 0.2),
                    ("run2", "full", 0.4),
                    ("run3", "sampled", 0.9),
                ]
            ):
                run_dir = root / run_id
                run_dir.mkdir()
                (run_dir / "metrics.json").write_text(
                    json.dumps(
                        {
                            "method_id": "sasrec",
                            "dataset": "toy",
                            "task": "sdsr",
                            "protocol": protocol,
                            "seed": idx,
                            "eval_input_type": "candidate_scores",
                            "metrics": {"recall@10": recall, "ndcg@10": recall / 2},
                        }
                    ),
                    encoding="utf-8",
                )

            rows = aggregate_runs(root)

        full = [row for row in rows if row["protocol"] == "full"][0]
        sampled = [row for row in rows if row["protocol"] == "sampled"][0]
        self.assertEqual(full["num_runs"], 2)
        self.assertAlmostEqual(full["recall@10_mean"], 0.3)
        self.assertGreater(full["recall@10_std"], 0)
        self.assertEqual(sampled["num_runs"], 1)
        self.assertAlmostEqual(sampled["recall@10_mean"], 0.9)


if __name__ == "__main__":
    unittest.main()
