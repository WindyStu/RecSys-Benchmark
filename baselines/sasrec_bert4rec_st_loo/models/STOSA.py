import torch
import torch.nn as nn
import torch.nn.functional as F

from models.subs.attn_encoders import SelfAttentionEncoder
from utils.metrics import cal_norm_mask
from utils.misc import init_weights


class STOSA(torch.nn.Module):
    """ Stochastic self-attention model with diagonal Gaussian item states. """
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.device = args.device
        self.d_latent = args.d_latent
        self.n_item = args.n_item
        self.idx_pad = args.idx_pad
        self.len_trim = args.len_trim
        self.bs = args.bs
        self.eval_mode = getattr(args, 'eval_mode', 'sampled')
        self.margin = getattr(args, 'margin', 0.0)
        self.eps = 1e-8

        self.emb_i_mean = nn.Embedding(self.n_item + 1, self.d_latent, padding_idx=0)
        self.emb_i_cov = nn.Embedding(self.n_item + 1, self.d_latent, padding_idx=0)
        self.emb_pos_mean = nn.Embedding(args.len_trim + 1, args.d_latent, padding_idx=0)
        self.emb_pos_cov = nn.Embedding(args.len_trim + 1, args.d_latent, padding_idx=0)

        self.dropout = nn.Dropout(args.dropout)
        self.attn_mean = SelfAttentionEncoder(args.d_latent, args.n_attn, args.n_head, args.dropout, args.len_trim)
        self.attn_cov = SelfAttentionEncoder(args.d_latent, args.n_attn, args.n_head, args.dropout, args.len_trim)
        self.norm_mean = nn.LayerNorm(self.d_latent)
        self.norm_cov = nn.LayerNorm(self.d_latent)

        self.apply(init_weights)

    def forward(self, seq, pos, mask_seq):
        h_mean = self.dropout(self.emb_i_mean(seq) + self.emb_pos_mean(pos))
        h_cov = self.dropout(F.softplus(self.emb_i_cov(seq) + self.emb_pos_cov(pos)) + self.eps)

        h_mean = self.norm_mean(self.attn_mean(h_mean, mask_seq))
        h_cov = F.softplus(self.norm_cov(self.attn_cov(h_cov, mask_seq))) + self.eps
        return h_mean, h_cov

    def wasserstein_distance(self, mean_a, cov_a, mean_b, cov_b):
        mean_dist = (mean_a - mean_b).pow(2).sum(-1)
        cov_dist = (torch.sqrt(cov_a + self.eps) - torch.sqrt(cov_b + self.eps)).pow(2).sum(-1)
        return mean_dist + cov_dist

    def cal_rec_loss_st(self, h_mean, h_cov, gt, gt_neg, mask_seq):
        e_gt_mean, e_gt_cov = self.emb_i_mean(gt), F.softplus(self.emb_i_cov(gt)) + self.eps
        e_neg_mean = self.emb_i_mean(gt_neg.squeeze(-1))
        e_neg_cov = F.softplus(self.emb_i_cov(gt_neg.squeeze(-1))) + self.eps

        dist_pos = self.wasserstein_distance(h_mean, h_cov, e_gt_mean, e_gt_cov)
        dist_neg = self.wasserstein_distance(h_mean, h_cov, e_neg_mean, e_neg_cov)
        loss = F.softplus(dist_pos - dist_neg + self.margin)
        return (loss * cal_norm_mask(mask_seq).squeeze(-1)).sum(-1).mean()

    def cal_rec_loss_dt(self, h_mean, h_cov, gt, gt_neg, mask_seq_a, mask_seq_b):
        e_gt_mean, e_gt_cov = self.emb_i_mean(gt), F.softplus(self.emb_i_cov(gt)) + self.eps
        e_neg_mean = self.emb_i_mean(gt_neg.squeeze(-1))
        e_neg_cov = F.softplus(self.emb_i_cov(gt_neg.squeeze(-1))) + self.eps

        dist_pos = self.wasserstein_distance(h_mean, h_cov, e_gt_mean, e_gt_cov)
        dist_neg = self.wasserstein_distance(h_mean, h_cov, e_neg_mean, e_neg_cov)
        loss = F.softplus(dist_pos - dist_neg + self.margin)
        loss_a = (loss * cal_norm_mask(mask_seq_a).squeeze(-1)).sum(-1).mean()
        loss_b = (loss * cal_norm_mask(mask_seq_b).squeeze(-1)).sum(-1).mean()
        return loss_a + loss_b

    def cal_rank_st(self, h_mean_last, h_cov_last, gt, gt_mtc):
        gt_mean = self.emb_i_mean(gt.squeeze(1))
        gt_cov = F.softplus(self.emb_i_cov(gt.squeeze(1))) + self.eps
        dist_gt = self.wasserstein_distance(h_mean_last, h_cov_last, gt_mean, gt_cov).unsqueeze(1)

        if self.eval_mode == 'full':
            item_mean = self.emb_i_mean.weight.unsqueeze(0)
            item_cov = (F.softplus(self.emb_i_cov.weight) + self.eps).unsqueeze(0)
            dist_all = self.wasserstein_distance(h_mean_last.unsqueeze(1), h_cov_last.unsqueeze(1), item_mean, item_cov)
            return (dist_all.lt(dist_gt) & gt_mtc.bool()).sum(-1).add(1).tolist()

        mtc_mean = self.emb_i_mean(gt_mtc)
        mtc_cov = F.softplus(self.emb_i_cov(gt_mtc)) + self.eps
        dist_mtc = self.wasserstein_distance(h_mean_last.unsqueeze(1), h_cov_last.unsqueeze(1), mtc_mean, mtc_cov)
        return dist_mtc.lt(dist_gt).sum(-1).add(1).tolist()

    def cal_rank_dt(self, h_mean_last, h_cov_last, gt, gt_mtc, mask_gt_a, mask_gt_b):
        gt_mean = self.emb_i_mean(gt.squeeze(1))
        gt_cov = F.softplus(self.emb_i_cov(gt.squeeze(1))) + self.eps
        dist_gt = self.wasserstein_distance(h_mean_last, h_cov_last, gt_mean, gt_cov).unsqueeze(1)

        if self.eval_mode == 'full':
            item_mean = self.emb_i_mean.weight.unsqueeze(0)
            item_cov = (F.softplus(self.emb_i_cov.weight) + self.eps).unsqueeze(0)
            dist_all = self.wasserstein_distance(h_mean_last.unsqueeze(1), h_cov_last.unsqueeze(1), item_mean, item_cov)
            ranks = (dist_all.lt(dist_gt) & gt_mtc.bool()).sum(-1).add(1)
        else:
            mtc_mean = self.emb_i_mean(gt_mtc)
            mtc_cov = F.softplus(self.emb_i_cov(gt_mtc)) + self.eps
            dist_mtc = self.wasserstein_distance(h_mean_last.unsqueeze(1), h_cov_last.unsqueeze(1), mtc_mean, mtc_cov)
            ranks = dist_mtc.lt(dist_gt).sum(-1).add(1)

        ranks_a = ranks[mask_gt_a.squeeze(-1) == 1].tolist()
        ranks_b = ranks[mask_gt_b.squeeze(-1) == 1].tolist()
        return ranks_a, ranks_b
