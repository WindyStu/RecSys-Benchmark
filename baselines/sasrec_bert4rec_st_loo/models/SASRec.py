import torch
import torch.nn as nn

from models.subs.attn_encoders import SelfAttentionEncoder
from utils.misc import init_weights
from utils.metrics import cal_norm_mask


class SASRec(torch.nn.Module):
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

        # model
        self.emb_i = nn.Embedding(self.n_item + 1, self.d_latent, padding_idx=0)
        self.emb_pos = nn.Embedding(args.len_trim + 1, args.d_latent, padding_idx=0)

        self.dropout = nn.Dropout(args.dropout)

        self.attn = SelfAttentionEncoder(args.d_latent, args.n_attn, args.n_head, args.dropout, args.len_trim)

        self.norm_attn = nn.LayerNorm(self.d_latent)

        self.CELoss = nn.CrossEntropyLoss(reduction='none', ignore_index=0)

        self.apply(init_weights)

    def forward(self, seq, pos, mask_seq):
        h = self.dropout(self.emb_i(seq) + self.emb_pos(pos))
        h = self.attn(h, mask_seq)
        h = self.norm_attn(h)
        return h

    def cal_rec_loss_st(self, h_seq, gt, mask_seq):
        """ calculate recommendation loss in single-target setting """
        logits = h_seq.matmul(self.emb_i.weight.t()).view(-1, self.n_item + 1)
        loss = (self.CELoss(logits,
                            gt.view(-1)).view(-1, self.len_trim) * cal_norm_mask(mask_seq).squeeze(-1)).sum(-1).mean()
        return loss

    def cal_rec_loss_dt(self, h_seq, gt, mask_seq_a, mask_seq_b):
        """ calculate recommendation loss in dual-target setting """
        logits = h_seq.matmul(self.emb_i.weight.t()).view(-1, self.n_item + 1)
        loss = self.CELoss(logits, gt.view(-1)).view(-1, self.len_trim)

        loss_a = (loss * cal_norm_mask(mask_seq_a).squeeze(-1)).sum(-1).mean()
        loss_b = (loss * cal_norm_mask(mask_seq_b).squeeze(-1)).sum(-1).mean()
        return loss_a + loss_b

    def cal_rank_st(self, h_last, gt, gt_mtc):
        """ calculate recommendation rank in single-target setting """
        if self.eval_mode == 'full':
            logits_all = h_last.matmul(self.emb_i.weight.t())
            logits_gt = logits_all.gather(1, gt)
            ranks = (logits_all.gt(logits_gt) & gt_mtc.bool()).sum(-1).add(1).tolist()
            return ranks

        e_gt, e_mtc = self.emb_i(gt),  self.emb_i(gt_mtc)
        logits_gt = (h_last * e_gt.squeeze(1)).sum(-1, keepdims=True)
        logits_mtc = (h_last.unsqueeze(1) * e_mtc).sum(-1)

        ranks = (logits_mtc - logits_gt).gt(0).sum(-1).add(1).tolist()
        return ranks

    def cal_rank_dt(self, h_last, gt, gt_mtc, mask_gt_a, mask_gt_b):
        """ calculate recommendation rank in dual-target setting """
        if self.eval_mode == 'full':
            logits_all = h_last.matmul(self.emb_i.weight.t())
            logits_gt = logits_all.gather(1, gt)
            ranks = (logits_all.gt(logits_gt) & gt_mtc.bool()).sum(-1).add(1)
            ranks_a = ranks[mask_gt_a.squeeze(-1) == 1].tolist()
            ranks_b = ranks[mask_gt_b.squeeze(-1) == 1].tolist()
            return ranks_a, ranks_b

        e_gt, e_mtc = self.emb_i(gt),  self.emb_i(gt_mtc)
        logits_gt = (h_last * e_gt.squeeze(1)).sum(-1, keepdims=True)
        logits_mtc = (h_last.unsqueeze(1) * e_mtc).sum(-1)

        ranks = (logits_mtc - logits_gt).gt(0).sum(-1).add(1)
        ranks_a = ranks[mask_gt_a.squeeze(-1) == 1].tolist()
        ranks_b = ranks[mask_gt_b.squeeze(-1) == 1].tolist()
        return ranks_a, ranks_b
