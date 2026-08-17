import tempfile
import unittest
from pathlib import Path

from recsys_benchmark.data.sanitize import copy_sanitized_tree, should_exclude_path


class SanitizeTest(unittest.TestCase):
    def test_excludes_generated_large_and_private_paths(self):
        self.assertTrue(should_exclude_path(Path(".git/config")))
        self.assertTrue(should_exclude_path(Path("logs/train.log")))
        self.assertTrue(should_exclude_path(Path("ckpt/model.pt")))
        self.assertTrue(should_exclude_path(Path("data/Beauty.inter.json")))
        self.assertTrue(should_exclude_path(Path("data/map_item.txt")))
        self.assertTrue(should_exclude_path(Path("data/afk/afk_50_preprocessed.txt")))
        self.assertTrue(should_exclude_path(Path("Dataset/1/toy_log_file_final")))
        self.assertTrue(should_exclude_path(Path(".claude/settings.local.json")))
        self.assertTrue(should_exclude_path(Path("docs/superpowers/spec.md")))
        self.assertFalse(should_exclude_path(Path("data/prepare_amazon.py")))
        self.assertFalse(should_exclude_path(Path("models/data/evaluation.py")))

    def test_copy_sanitized_tree_redacts_secret_like_text(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            src = Path(tmpdir) / "src"
            dst = Path(tmpdir) / "dst"
            src.mkdir()
            (src / "aliyun_text_emb.py").write_text(
                'DEFAULT_API_KEY = "sk-test-secret"\nDEFAULT_BASE_URL = "https://private.example/v1"\nprint("ok")\n',
                encoding="utf-8",
            )
            (src / "model.pt").write_text("large binary placeholder", encoding="utf-8")

            copied = copy_sanitized_tree(src, dst)

            redacted = (dst / "aliyun_text_emb.py").read_text(encoding="utf-8")

        self.assertEqual(copied["copied_files"], 1)
        self.assertIn("REPLACE_WITH_ENV_VAR", redacted)
        self.assertIn("REPLACE_WITH_BASE_URL", redacted)
        self.assertFalse((dst / "model.pt").exists())


if __name__ == "__main__":
    unittest.main()
