import json
import pickle
import sys
import tempfile
import types
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "tri_cdr"))
sys.modules.setdefault("ipdb", types.SimpleNamespace(set_trace=lambda: None))

from utils import data_partition, evaluate_SASRec
import train_sasrec


class FakeTriCDRModel:
    def __init__(self):
        self.candidates = None

    def predict(self, user_ids, seq_mix, seq_source, seq_target, item_idx):
        candidates = item_idx[0].tolist()
        self.candidates = candidates
        scores = torch.arange(len(candidates), 0, -1, dtype=torch.float32).unsqueeze(0)
        feat = torch.zeros((1, 4), dtype=torch.float32)
        return scores, feat, feat, feat


class CaptureEmbedding(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ids = None

    def forward(self, ids):
        self.ids = ids.detach().cpu().tolist()
        return torch.ones((len(ids), 4), dtype=torch.float32)


class FakeSASRecModel:
    def __init__(self):
        self.item_emb = CaptureEmbedding()
        self.input_seq = None

    def eval(self):
        pass

    def log2feats(self, input_seq):
        self.input_seq = input_seq.tolist()
        maxlen = input_seq.shape[1]
        return torch.ones((1, maxlen, 4), dtype=torch.float32)


class TriCDRUtilsTests(unittest.TestCase):
    def _write_pickle(self, path, value):
        with open(path, "wb") as f:
            pickle.dump(value, f)

    def _write_dataset(self, root):
        data_dir = Path(root) / "toy" / "Source_Target"
        data_dir.mkdir(parents=True)

        self._write_pickle(data_dir / "Source_log_file_final.pkl", {10: [1, 2]})
        self._write_pickle(data_dir / "Target_log_file_final.pkl", {10: [4, 5, 6]})
        self._write_pickle(data_dir / "mix_log_file_final.pkl", {10: [1, 4, 2, 5, 6]})
        self._write_pickle(data_dir / "item_index_mix.pkl", {"1": 1, "2": 2, "4": 4, "5": 5, "6": 6})
        self._write_pickle(data_dir / "user_index_overleap.pkl", {10: True})

        np.save(data_dir / "item_index_Source.npy", np.array(["1", "2"]))
        np.save(data_dir / "item_index_Target.npy", np.array(["4", "5", "6", "7", "8"]))

        with open(data_dir / "meta.json", "w") as f:
            json.dump({"interval": 3, "itemnum": 8}, f)

    def test_data_partition_compacts_sparse_user_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            self._write_dataset(tmp)

            dataset = data_partition("Source", "Target", "toy", maxlen=5, data_root=tmp)
            user_train_mix, user_train_source, user_train_target = dataset[:3]
            user_valid_target, user_test_target = dataset[3:5]
            usernum = dataset[7]

        self.assertEqual(usernum, 1)
        self.assertEqual(sorted(user_train_mix.keys()), [1])
        self.assertEqual(user_train_source[1], [1, 2])
        self.assertEqual(user_train_target[1], [4])
        self.assertEqual(user_valid_target[1], [5])
        self.assertEqual(user_test_target[1], [6])

    def test_evaluate_uses_full_unseen_target_domain_candidates(self):
        dataset = [
            {1: [1, 4, 2]},
            {1: [1, 2]},
            {1: [4]},
            {1: [5]},
            {1: [6]},
            {},
            {},
            1,
            8,
            3,
        ]
        args = SimpleNamespace(maxlen=5, device="cpu", num_samples=3)
        model = FakeTriCDRModel()

        evaluate_SASRec(model, dataset, args)

        self.assertEqual(model.candidates, [6, 7, 8])

    def test_sasrec_validation_uses_second_to_last_item_not_test_item(self):
        user_seqs = {1: [1, 2, 3, 4]}
        model = FakeSASRecModel()

        train_sasrec.evaluate(
            model,
            user_seqs,
            all_users=[1],
            item_list=[1, 2, 3, 4, 5, 6],
            maxlen=5,
            device="cpu",
        )

        self.assertEqual(model.item_emb.ids[0], 3)
        self.assertEqual(model.input_seq[0][-2:], [1, 2])


if __name__ == "__main__":
    unittest.main()
