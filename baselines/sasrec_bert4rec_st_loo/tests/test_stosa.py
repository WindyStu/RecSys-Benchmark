import unittest
from types import SimpleNamespace

import torch

from models.STOSA import STOSA


def make_args(n_item=5, eval_mode="full"):
    return SimpleNamespace(
        device=torch.device("cpu"),
        d_latent=4,
        n_item=n_item,
        idx_pad=0,
        len_trim=3,
        bs=2,
        dropout=0.0,
        n_attn=1,
        n_head=1,
        eval_mode=eval_mode,
        margin=0.0,
    )


class STOSATest(unittest.TestCase):
    def test_forward_returns_mean_and_covariance_sequence_embeddings(self):
        model = STOSA(make_args())
        seq = torch.tensor([[1, 2, 3], [0, 1, 4]])
        pos = torch.tensor([[1, 2, 3], [0, 1, 2]])
        mask_seq = torch.tensor([[[1], [1], [1]], [[0], [1], [1]]])

        mean_seq, cov_seq = model(seq, pos, mask_seq)

        self.assertEqual(mean_seq.shape, (2, 3, 4))
        self.assertEqual(cov_seq.shape, (2, 3, 4))
        self.assertTrue(torch.all(cov_seq >= 0))

    def test_sampled_training_loss_is_finite(self):
        model = STOSA(make_args(eval_mode="sampled"))
        seq = torch.tensor([[1, 2, 3], [0, 1, 4]])
        pos = torch.tensor([[1, 2, 3], [0, 1, 2]])
        mask_seq = torch.tensor([[[1], [1], [1]], [[0], [1], [1]]])
        gt = torch.tensor([[2, 3, 4], [0, 4, 5]])
        gt_neg = torch.tensor([[[5], [4], [2]], [[0], [2], [3]]])

        h_mean, h_cov = model(seq, pos, mask_seq)
        loss = model.cal_rec_loss_st(h_mean, h_cov, gt, gt_neg, mask_seq)

        self.assertTrue(torch.isfinite(loss))

    def test_full_rank_uses_lower_distance_as_better(self):
        model = STOSA(make_args(n_item=4))
        with torch.no_grad():
            model.emb_i_mean.weight.zero_()
            model.emb_i_cov.weight.fill_(0.5)
            model.emb_i_mean.weight[2] = torch.tensor([0.0, 0.0, 0.0, 0.0])
            model.emb_i_mean.weight[3] = torch.tensor([0.5, 0.0, 0.0, 0.0])
            model.emb_i_mean.weight[4] = torch.tensor([3.0, 0.0, 0.0, 0.0])

        h_mean = torch.zeros(1, 4)
        h_cov = torch.full((1, 4), 0.5)
        gt = torch.tensor([[2]])
        candidate_mask = torch.tensor([[0, 0, 0, 1, 1]])

        self.assertEqual(model.cal_rank_st(h_mean, h_cov, gt, candidate_mask), [1])


if __name__ == "__main__":
    unittest.main()
