import json
import tempfile
import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))

from metrics import compute_metrics
from sdsr_data import prepare_sdsr_domain
from generate_training_batches import Train_instance


class MetricsTest(unittest.TestCase):
    def test_compute_metrics_reports_requested_cutoffs(self):
        predictions = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10], [9, 8, 7, 6, 5, 4]]
        labels = [[1], [4]]

        metrics = compute_metrics(predictions, labels, cutoffs=(5, 10))

        self.assertAlmostEqual(metrics["recall@5"], 0.5)
        self.assertAlmostEqual(metrics["recall@10"], 1.0)
        self.assertAlmostEqual(metrics["ndcg@5"], 0.5)
        self.assertGreater(metrics["ndcg@10"], metrics["ndcg@5"])


class SdsrDataTest(unittest.TestCase):
    def test_prepare_sdsr_domain_writes_leave_one_out_instances(self):
        with tempfile.TemporaryDirectory() as tmp:
            data_dir = Path(tmp) / "Demo"
            out_dir = Path(tmp) / "work" / "Demo"
            data_dir.mkdir()
            (data_dir / "Demo.inter.json").write_text(
                json.dumps({"0": [1, 2, 3, 4, 5], "1": [2, 3, 4, 5, 6, 7]}),
                encoding="utf-8",
            )
            (data_dir / "Demo.item.json").write_text(
                json.dumps({str(i): {"title": f"item {i}"} for i in range(8)}),
                encoding="utf-8",
            )

            summary = prepare_sdsr_domain(
                data_dir=data_dir,
                output_dir=out_dir,
                dataset="Demo",
                seq_len=4,
                min_seq_len=5,
                train_sample_seg_cnt=2,
                seed=7,
                force=True,
            )

            self.assertEqual(summary.user_num, 2)
            self.assertEqual(summary.item_num, 8)
            self.assertTrue((out_dir / "item_node_num.txt").exists())
            self.assertTrue((out_dir / "train_instances_0").exists())
            self.assertTrue((out_dir / "train_instances_1").exists())
            self.assertEqual((out_dir / "test_instances").read_text(encoding="utf-8").count("\n"), 2)
            self.assertIn("|7", (out_dir / "test_instances").read_text(encoding="utf-8"))

    def test_training_batch_cache_uses_requested_paths(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            train_prefix = tmp_path / "train_instances"
            (tmp_path / "train_instances_0").write_text("0|-1,-1,1|2\n", encoding="utf-8")
            history_cache = tmp_path / "his_matrix.pt"
            labels_cache = tmp_path / "labels.pt"

            Train_instance().get_training_data(
                str(train_prefix),
                1,
                3,
                str(history_cache),
                str(labels_cache),
            )

            self.assertTrue(history_cache.exists())
            self.assertTrue(labels_cache.exists())


if __name__ == "__main__":
    unittest.main()
