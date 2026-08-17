import importlib.util
import tempfile
import unittest
from pathlib import Path


VQREC_DIR = Path(__file__).resolve().parents[1]
PREFLIGHT_PATH = VQREC_DIR / "vqrec_preflight.py"

spec = importlib.util.spec_from_file_location("vqrec_preflight", PREFLIGHT_PATH)
vqrec_preflight = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(vqrec_preflight)


HEADER = "user_id:token\titem_id_list:token_seq\titem_id:token\ttimestamp:float"


class VQRecPreflightTest(unittest.TestCase):
    def test_preflight_accepts_valid_inter_files(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            for suffix in ("train", "valid", "test"):
                (data_dir / f"Toy.{suffix}.inter").write_text(
                    "\n".join([HEADER, "1\t10 11\t12\t3.0"])
                )

            vqrec_preflight.validate_prepared_inter_files(str(data_dir), "Toy", 50)

    def test_preflight_rejects_empty_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            (data_dir / "Toy.train.inter").write_text(
                "\n".join([HEADER, "1\t\t10\t1.0"])
            )

            with self.assertRaisesRegex(ValueError, "empty item_id_list"):
                vqrec_preflight.validate_prepared_inter_file(
                    str(data_dir / "Toy.train.inter"), 50
                )

    def test_preflight_rejects_history_over_max_length(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            data_dir = Path(tmpdir)
            history = " ".join(str(i) for i in range(51))
            (data_dir / "Toy.train.inter").write_text(
                "\n".join([HEADER, f"1\t{history}\t52\t53.0"])
            )

            with self.assertRaisesRegex(ValueError, "exceeds MAX_ITEM_LIST_LENGTH"):
                vqrec_preflight.validate_prepared_inter_file(
                    str(data_dir / "Toy.train.inter"), 50
                )


if __name__ == "__main__":
    unittest.main()
