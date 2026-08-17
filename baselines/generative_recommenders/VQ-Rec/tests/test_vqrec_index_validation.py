import importlib.util
import unittest
from pathlib import Path

import torch


VQREC_DIR = Path(__file__).resolve().parents[1]
VALIDATION_PATH = VQREC_DIR / "vqrec_index_validation.py"

spec = importlib.util.spec_from_file_location(
    "vqrec_index_validation", VALIDATION_PATH
)
vqrec_index_validation = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(vqrec_index_validation)


class VQRecIndexValidationTest(unittest.TestCase):
    def test_forward_indices_accept_valid_batch(self):
        vqrec_index_validation.validate_forward_indices(
            item_seq=torch.tensor([[1, 2, 0]]),
            item_seq_len=torch.tensor([2]),
            pq_codes=torch.zeros((3, 32), dtype=torch.long),
            max_position_embeddings=3,
            code_embedding_rows=8224,
        )

    def test_forward_indices_reject_sequence_wider_than_position_embeddings(self):
        with self.assertRaisesRegex(ValueError, "item_seq width"):
            vqrec_index_validation.validate_forward_indices(
                item_seq=torch.tensor([[1, 2, 3, 4]]),
                item_seq_len=torch.tensor([4]),
                pq_codes=torch.zeros((5, 32), dtype=torch.long),
                max_position_embeddings=3,
                code_embedding_rows=8224,
            )

    def test_forward_indices_reject_empty_histories(self):
        with self.assertRaisesRegex(ValueError, "item_seq_len"):
            vqrec_index_validation.validate_forward_indices(
                item_seq=torch.tensor([[0, 0, 0]]),
                item_seq_len=torch.tensor([0]),
                pq_codes=torch.zeros((3, 32), dtype=torch.long),
                max_position_embeddings=3,
                code_embedding_rows=8224,
            )

    def test_forward_indices_reject_item_ids_outside_pq_table(self):
        with self.assertRaisesRegex(ValueError, "item_seq"):
            vqrec_index_validation.validate_forward_indices(
                item_seq=torch.tensor([[1, 3, 0]]),
                item_seq_len=torch.tensor([2]),
                pq_codes=torch.zeros((3, 32), dtype=torch.long),
                max_position_embeddings=3,
                code_embedding_rows=8224,
            )

    def test_forward_indices_reject_pq_codes_outside_embedding_table(self):
        pq_codes = torch.zeros((3, 32), dtype=torch.long)
        pq_codes[1, 0] = 8224

        with self.assertRaisesRegex(ValueError, "pq_codes"):
            vqrec_index_validation.validate_forward_indices(
                item_seq=torch.tensor([[1, 2, 0]]),
                item_seq_len=torch.tensor([2]),
                pq_codes=pq_codes,
                max_position_embeddings=3,
                code_embedding_rows=8224,
            )

    def test_ce_targets_reject_labels_outside_logits(self):
        with self.assertRaisesRegex(ValueError, "target"):
            vqrec_index_validation.validate_ce_targets(
                targets=torch.tensor([0, 3]),
                num_classes=3,
                field_name="item_id",
            )


if __name__ == "__main__":
    unittest.main()
