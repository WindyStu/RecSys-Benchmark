import tempfile
import sys
import unittest
import json
from pathlib import Path

from recsys_benchmark.adapters.command import CommandAdapter


class CommandAdapterTest(unittest.TestCase):
    def test_dry_run_renders_command_and_writes_artifact_metadata(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            adapter = CommandAdapter(
                {
                    "method": {
                        "method_id": "toy",
                        "source": str(root),
                        "commands": {
                            "train": ["python", "main.py", "--dataset", "{dataset.dataset_id}", "--seed", "{seed}"]
                        },
                    },
                    "dataset": {"dataset_id": "toy_data"},
                    "seed": 7,
                    "output_dir": str(root / "outputs"),
                    "dry_run": True,
                }
            )

            result = adapter.train()

        self.assertEqual(result["returncode"], 0)
        self.assertEqual(result["command"], ["python", "main.py", "--dataset", "toy_data", "--seed", "7"])
        self.assertEqual(result["mode"], "dry_run")

    def test_subprocess_stdout_and_stderr_are_saved_in_stage_artifact(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            source = root / "source"
            source.mkdir()
            (source / "run.py").write_text(
                "import sys\nprint('hello stdout')\nprint('hello stderr', file=sys.stderr)\n",
                encoding="utf-8",
            )
            adapter = CommandAdapter(
                {
                    "method": {
                        "method_id": "toy",
                        "source": str(source),
                        "commands": {"train": [sys.executable, "run.py"]},
                    },
                    "dataset": {"dataset_id": "toy"},
                    "output_dir": str(root / "outputs"),
                }
            )

            result = adapter.train()
            artifact = json.loads((root / "outputs" / "logs" / "train.json").read_text(encoding="utf-8"))

        self.assertEqual(result["returncode"], 0)
        self.assertIn("hello stdout", artifact["stdout"])
        self.assertIn("hello stderr", artifact["stderr"])


if __name__ == "__main__":
    unittest.main()
