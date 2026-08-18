import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from recsys_benchmark.adapters.abxi import ABXIAdapter
from recsys_benchmark.adapters.dual_domain import parse_dual_domain_log
from recsys_benchmark.adapters.merit import MERITAdapter
from recsys_benchmark.config.loader import load_experiment_config
from recsys_benchmark.config.readiness import inspect_method_readiness
from scripts.prepare_cdsr_sequence import prepare_cdsr_sequence


ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"


class DualDomainRankerTest(unittest.TestCase):
    def test_sequence_bridge_remaps_items_by_domain(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "asc"
            source.mkdir()
            (source / "asc.inter.json").write_text(json.dumps({"0": [0, 2, 1, 3]}), encoding="utf-8")
            (source / "map_item.txt").write_text(
                json.dumps({"sa": [0, 0], "ca": [1, 1], "sb": [2, 0], "cb": [3, 1]}), encoding="utf-8"
            )
            (source / "map_user.txt").write_text(json.dumps({"u": 0}), encoding="utf-8")

            result = prepare_cdsr_sequence(source, root / "target", "asc", len_max=50)
            out = Path(result["out_dir"])
            item_map = json.loads((out / "map_item_50.txt").read_text(encoding="utf-8"))
            sequence = (out / "asc_50_preprocessed.txt").read_text(encoding="utf-8").strip()

        self.assertEqual(item_map, {"sa": [1, 0], "ca": [3, 1], "sb": [2, 0], "cb": [4, 1]})
        self.assertEqual(sequence, "0 1|0 2|1 3|2 4|3")

    def test_dual_domain_log_parser_emits_macro_and_domain_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "run.log"
            path.write_text(
                "| 0.1000 | 0.2000 | 0.0500 | 0.0700 | 0.0300 | 0.3000 | 0.4000 | 0.1500 | 0.1700 | 0.1300 |\n",
                encoding="utf-8",
            )
            metrics = parse_dual_domain_log(path, ("Sports", "Clothing"))

        self.assertAlmostEqual(metrics["recall@5"], 0.2)
        self.assertAlmostEqual(metrics["ndcg@10"], 0.12)
        self.assertAlmostEqual(metrics["domainrecall@10:Sports"], 0.2)
        self.assertAlmostEqual(metrics["domainndcg@5:Clothing"], 0.15)
        self.assertAlmostEqual(metrics["crossdomaintransfergap@5"], 0.2)

    def test_configs_render_and_are_adapter_ready(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ, {"RECSYS_DATA_ROOT": str(Path(tmpdir) / "data")}, clear=False
        ):
            for method_id, adapter_class in (("abxi", ABXIAdapter), ("merit", MERITAdapter)):
                with self.subTest(method_id=method_id):
                    config = load_experiment_config(
                        CONFIGS / "experiments" / f"asc_{method_id}.yaml", config_root=CONFIGS
                    )
                    config["dry_run"] = True
                    config["output_dir"] = str(Path(tmpdir) / "runs" / method_id)
                    adapter = adapter_class(config)

                    prepare = adapter.prepare()
                    train = adapter.train()
                    readiness = inspect_method_readiness(CONFIGS / "methods" / f"{method_id}.yaml")

                    self.assertIn("prepare_cdsr_sequence.py", " ".join(prepare["command"]))
                    self.assertIn("asc", train["command"])
                    self.assertEqual(readiness["computed_status"], "adapter-ready")


if __name__ == "__main__":
    unittest.main()
