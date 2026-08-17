import json
import tempfile
import unittest
from pathlib import Path

from scripts.prepare_sdsr_data import prepare_single_domain


class PrepareSdsrDataTest(unittest.TestCase):
    def test_prepare_single_domain_writes_project_dataset_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source" / "Beauty"
            target = root / "target"
            source.mkdir(parents=True)
            (source / "Beauty.inter.json").write_text(
                json.dumps({"u2": [3, 4], "u1": [0, 1, 2]}),
                encoding="utf-8",
            )

            stats = prepare_single_domain(
                source_root=root / "source",
                target_root=target,
                domain="Beauty",
                out_name="abeauty",
                len_max=50,
            )

            out_dir = target / "abeauty"
            self.assertEqual(stats["n_user"], 2)
            self.assertEqual(stats["n_item"], 5)
            self.assertTrue((out_dir / "map_user.txt").exists())
            self.assertTrue((out_dir / "map_item.txt").exists())
            self.assertEqual(
                (out_dir / "abeauty_50_preprocessed.txt").read_text(encoding="utf-8").splitlines(),
                ["0 1|0 2|1 3|2", "1 4|0 5|1"],
            )


if __name__ == "__main__":
    unittest.main()
