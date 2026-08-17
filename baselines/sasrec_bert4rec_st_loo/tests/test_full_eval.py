import unittest
from types import SimpleNamespace

import torch

from models.BERT4Rec import BERT4Rec


def make_args(n_item):
    return SimpleNamespace(
        device=torch.device("cpu"),
        d_latent=2,
        n_item=n_item,
        idx_pad=0,
        len_trim=2,
        bs=1,
        dropout=0.0,
        n_attn=1,
        n_head=1,
        eval_mode="full",
    )


class FullEvalTest(unittest.TestCase):
    def test_single_target_full_rank_ignores_seen_items(self):
        model = BERT4Rec(make_args(n_item=4))
        with torch.no_grad():
            model.emb_i.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0],
                        [0.9, 0.0],
                        [0.5, 0.0],
                        [0.7, 0.0],
                        [0.1, 0.0],
                    ]
                )
            )

        h_last = torch.tensor([[1.0, 0.0]])
        gt = torch.tensor([[2]])
        candidate_mask = torch.tensor([[0, 0, 0, 1, 1]])

        self.assertEqual(model.cal_rank_st(h_last, gt, candidate_mask), [2])

    def test_dual_target_full_rank_compares_only_positive_domain_candidates(self):
        model = BERT4Rec(make_args(n_item=5))
        with torch.no_grad():
            model.emb_i.weight.copy_(
                torch.tensor(
                    [
                        [0.0, 0.0],
                        [0.9, 0.0],
                        [0.5, 0.0],
                        [0.1, 0.0],
                        [2.0, 0.0],
                        [1.5, 0.0],
                    ]
                )
            )

        h_last = torch.tensor([[1.0, 0.0]])
        gt = torch.tensor([[2]])
        candidate_mask = torch.tensor([[0, 1, 0, 1, 0, 0]])
        mask_gt_a = torch.tensor([[1]])
        mask_gt_b = torch.tensor([[0]])

        ranks_a, ranks_b = model.cal_rank_dt(h_last, gt, candidate_mask, mask_gt_a, mask_gt_b)

        self.assertEqual(ranks_a, [2])
        self.assertEqual(ranks_b, [])


if __name__ == "__main__":
    unittest.main()
