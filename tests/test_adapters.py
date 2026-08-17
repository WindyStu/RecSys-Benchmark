import tempfile
import unittest
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


if __name__ == "__main__":
    unittest.main()
