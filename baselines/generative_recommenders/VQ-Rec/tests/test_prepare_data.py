import importlib.util
import tempfile
import unittest
from pathlib import Path


VQREC_DIR = Path(__file__).resolve().parents[1]
PREPARE_DATA_PATH = VQREC_DIR / "prepare_data.py"

spec = importlib.util.spec_from_file_location("prepare_data", PREPARE_DATA_PATH)
prepare_data = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(prepare_data)


class PrepareDataTest(unittest.TestCase):
    def test_convert_to_recbole_inter_writes_atomic_header(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "Cell_Phones.inter"

            prepare_data.convert_to_recbole_inter({"7": [3, 9]}, str(output_path))

            lines = output_path.read_text().splitlines()
            self.assertEqual(
                lines[0],
                "user_id:token\titem_id_list:token_seq\titem_id:token\ttimestamp:float",
            )
            self.assertEqual(lines[1:], ["7\t3\t9\t2.0"])

    def test_convert_to_recbole_inter_truncates_history_to_max_item_list_length(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "Long.inter"
            item_ids = list(range(53))

            prepare_data.convert_to_recbole_inter({"1": item_ids}, str(output_path))

            last_line = output_path.read_text().splitlines()[-1]
            history = last_line.split("\t")[1].split()
            self.assertEqual(last_line.split("\t")[2], "52")
            self.assertEqual(len(history), prepare_data.DEFAULT_MAX_ITEM_LIST_LENGTH)
            self.assertEqual(history[0], "2")
            self.assertEqual(history[-1], "51")

    def test_create_splits_preserves_header_and_splits_only_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dst_dir = Path(tmpdir)
            prepare_data.convert_to_recbole_inter(
                {"1": [10, 11, 12], "2": [20, 21]}, str(dst_dir / "Toy.inter")
            )

            prepare_data._create_splits(str(dst_dir), "Toy")

            header = "user_id:token\titem_id_list:token_seq\titem_id:token\ttimestamp:float"
            train_lines = (dst_dir / "Toy.train.inter").read_text().splitlines()
            valid_lines = (dst_dir / "Toy.valid.inter").read_text().splitlines()
            test_lines = (dst_dir / "Toy.test.inter").read_text().splitlines()

            self.assertEqual(train_lines[0], header)
            self.assertEqual(valid_lines[0], header)
            self.assertEqual(test_lines[0], header)
            self.assertEqual(train_lines[1:], ["1\t10\t11\t2.0", "2\t20\t21\t2.0"])
            self.assertEqual(valid_lines[1:], ["1\t10 11\t12\t3.0", "2\t20\t21\t2.0"])
            self.assertEqual(test_lines[1:], ["1\t10 11\t12\t3.0", "2\t20\t21\t2.0"])


if __name__ == "__main__":
    unittest.main()
