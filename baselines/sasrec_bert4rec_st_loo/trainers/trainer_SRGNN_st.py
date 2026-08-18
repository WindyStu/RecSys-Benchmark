import time

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau
from tqdm import tqdm

from dataloaders.dataloader_SASRec_st import get_dataloader
from models.SR_GNN import SRGNN
from utils.metrics import cal_metrics


class Trainer(object):
    def __init__(self, args, noter):
        print('[info] Loading data')
        self.dl = get_dataloader(args)
        self.n_user, self.n_item = self.dl.dataset.get_stat()
        print('Done.\n')

        self.noter = noter
        self.device = args.device
        self.eval_mode = args.eval_mode

        self.model = SRGNN(args).to(args.device)
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.l2)
        self.scheduler_warmup = LinearLR(
            self.optimizer, start_factor=1e-5, end_factor=1.0, total_iters=args.n_warmup
        )
        self.scheduler = ReduceLROnPlateau(
            self.optimizer, mode='max', factor=args.lr_g, patience=args.lr_p
        )

        noter.log_msg(
            f'[info] model contains {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}'
            f' learnable parameters.'
        )

    def set_train(self):
        self.model.train()
        self.dl.dataset.set_train()

    def set_valid(self):
        self.model.eval()
        self.dl.dataset.set_valid()

    def set_test(self):
        self.model.eval()
        self.dl.dataset.set_test()

    def run_epoch(self, i_epoch):
        self.set_train()
        loss = 0.0
        n_examples = 0
        t0 = time.time()

        self.noter.log_msg(f'\n[epoch {i_epoch:>2}]')
        for batch in tqdm(self.dl, desc='training', leave=False):
            loss_batch, batch_size = self.train_batch(batch)
            loss += loss_batch * batch_size
            n_examples += batch_size

        self.noter.log_train(loss / max(n_examples, 1), time.time() - t0)

        self.set_valid()
        ranks = []
        with torch.no_grad():
            for batch in tqdm(self.dl, desc='validating', leave=False):
                ranks.extend(self.evaluate_batch(batch))
        return cal_metrics(ranks)

    def run_test(self):
        self.set_test()
        ranks = []
        with torch.no_grad():
            for batch in tqdm(self.dl, desc='testing', leave=False):
                ranks.extend(self.evaluate_batch(batch))
        return cal_metrics(ranks)

    def train_batch(self, batch):
        seq, _, mask_seq, gt = map(lambda value: value.to(self.device), batch)
        targets = gt[:, -1]
        valid = targets.gt(0)
        if not valid.any():
            return 0.0, 0

        seq = seq[valid]
        targets = targets[valid]
        lengths = mask_seq[valid].squeeze(-1).sum(dim=1)

        self.optimizer.zero_grad()
        loss = self.model.calculate_loss(seq, lengths, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item(), int(valid.sum().item())

    def evaluate_batch(self, batch):
        seq, _, mask_seq, gt, candidates = map(lambda value: value.to(self.device), batch)
        targets = gt.squeeze(-1)
        valid = targets.gt(0)
        if not valid.any():
            return []

        seq = seq[valid]
        lengths = mask_seq[valid].squeeze(-1).sum(dim=1)
        targets = targets[valid]
        candidates = candidates[valid]
        scores = self.model.full_sort_predict(seq, lengths)
        target_scores = scores.gather(1, targets.unsqueeze(1))

        if self.eval_mode == 'full':
            ranks = (scores.gt(target_scores) & candidates.bool()).sum(dim=1).add(1)
        else:
            negative_scores = scores.gather(1, candidates)
            ranks = negative_scores.gt(target_scores).sum(dim=1).add(1)
        return ranks.tolist()
