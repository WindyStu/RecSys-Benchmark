import json
import tempfile
import textwrap
import unittest
from pathlib import Path

from recsys_benchmark.adapters.sasrec_bert4rec import SASRecBERT4RecAdapter
from recsys_benchmark.config.readiness import inspect_method_readiness


class SASRecFamilyAdapterTest(unittest.TestCase):
    def test_evaluate_parses_native_st_log_into_metrics_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            log_dir = root / "log"
            log_dir.mkdir()
            (log_dir / "sasrec-st-beauty-08-18.log").write_text(
                textwrap.dedent(
                    """
                    ---------- Experiment ended ----------
                    [Info] sasrec (data:beauty, cuda:cpu)
                    [ Info ] sasrec-st (1.0 min)
                          |                A/B                |
                          | hr5    | hr10   | ndcg5  | ndcg10 |  mrr   |
                    |  F  | 0.1000 | 0.2000 | 0.0500 | 0.0700 | 0.0300 |
                    """
                ),
                encoding="utf-8",
            )
            adapter = SASRecBERT4RecAdapter(
                {
                    "method": {
                        "method_id": "sasrec",
                        "source": str(root),
                        "native_model": "sasrec",
                        "native_metrics": {
                            "type": "sasrec_st_log",
                            "log_dir": "{method.source}/log",
                            "pattern": "sasrec-st-{dataset.dataset_id}-*.log",
                        },
                    },
                    "dataset": {"dataset_id": "beauty", "task": "sdsr"},
                    "seed": 3407,
                    "evaluation": {"protocol": "full"},
                    "output_dir": str(root / "run"),
                }
            )

            result = adapter.evaluate()
            record = json.loads((root / "run" / "metrics.json").read_text(encoding="utf-8"))

        self.assertEqual(result["status"], "evaluated")
        self.assertEqual(record["eval_input_type"], "native_metrics")
        self.assertAlmostEqual(record["metrics"]["recall@5"], 0.1)
        self.assertAlmostEqual(record["metrics"]["hitrate@10"], 0.2)
        self.assertAlmostEqual(record["metrics"]["ndcg@10"], 0.07)
        self.assertAlmostEqual(record["metrics"]["mrr@10"], 0.03)

    def test_readiness_accepts_native_metrics_contract_for_ranker(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            source.mkdir()
            config_path = root / "sasrec.yaml"
            config_path.write_text(
                textwrap.dedent(
                    f"""
                    method_id: sasrec
                    method_type: ranker
                    status: source-integrated
                    source: {source.as_posix()}
                    adapter: recsys_benchmark.adapters.sasrec_bert4rec.SASRecBERT4RecAdapter
                    commands:
                      train: [python, main_st.py, --m, sasrec]
                    native_metrics:
                      type: sasrec_st_log
                      log_dir: "{{method.source}}/log"
                      pattern: "sasrec-st-{{dataset.dataset_id}}-*.log"
                    """
                ),
                encoding="utf-8",
            )

            report = inspect_method_readiness(config_path)

        self.assertEqual(report["computed_status"], "adapter-ready")
        self.assertEqual(report["missing"], [])


if __name__ == "__main__":
    unittest.main()
