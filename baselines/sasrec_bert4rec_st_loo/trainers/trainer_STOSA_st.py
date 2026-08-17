import time
from tqdm import tqdm
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau

from models.STOSA import STOSA
from dataloaders.dataloader_BERT4Rec_st import get_dataloader
from utils.metrics import cal_metrics


class Trainer(object):
    def __init__(self, args, noter):
        print('[info] Loading data')
        self.dl = get_dataloader(args)
        self.n_user, self.n_item = self.dl.dataset.get_stat()
        print('Done.\n')

        self.noter = noter
        self.device = args.device

        self.model = STOSA(args).to(args.device)
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr, weight_decay=args.l2)
        self.scheduler_warmup = LinearLR(self.optimizer, start_factor=1e-5, end_factor=1., total_iters=args.n_warmup)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='max', factor=args.lr_g, patience=args.lr_p)

        noter.log_msg(f'[info] model contains {sum(p.numel() for p in self.model.parameters() if p.requires_grad)}'
                      f' learnable parameters.')

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
        loss_rec = 0.
        t0 = time.time()

        self.noter.log_msg(f'\n[epoch {i_epoch:>2}]')
        for batch in tqdm(self.dl, desc='training', leave=False):
            self.optimizer.zero_grad()
            loss_rec_batch = self.train_batch(batch)
            n_seq = batch[0].shape[0]
            loss_rec += (loss_rec_batch * n_seq)

        self.noter.log_train(loss_rec / self.n_user, time.time() - t0)

        self.set_valid()
        ranks = []
        with torch.no_grad():
            for batch in tqdm(self.dl, desc='validating', leave=False):
                ranks += self.evaluate_batch(batch)
        return cal_metrics(ranks)

    def run_test(self):
        self.set_test()
        ranks = []
        with torch.no_grad():
            for batch in tqdm(self.dl, desc='testing', leave=False):
                ranks += self.evaluate_batch(batch)
        return cal_metrics(ranks)

    def train_batch(self, batch):
        seq, pos, mask_seq, gt, gt_neg = map(lambda x: x.to(self.device), batch)
        h_mean, h_cov = self.model(seq, pos, mask_seq)
        loss_rec = self.model.cal_rec_loss_st(h_mean, h_cov, gt, gt_neg, mask_seq)
        loss_rec.backward()
        self.optimizer.step()
        return loss_rec.item()

    def evaluate_batch(self, batch):
        seq, pos, mask_seq, gt, gt_mtc = map(lambda x: x.to(self.device), batch)
        h_mean, h_cov = self.model(seq, pos, mask_seq)
        return self.model.cal_rank_st(h_mean[:, -1], h_cov[:, -1], gt, gt_mtc)
