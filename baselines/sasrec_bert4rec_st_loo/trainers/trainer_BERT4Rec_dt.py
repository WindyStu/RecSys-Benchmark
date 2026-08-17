import time
from tqdm import tqdm
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LinearLR, ReduceLROnPlateau

from models.BERT4Rec import BERT4Rec
from dataloaders.dataloader_BERT4Rec_dt import get_dataloader
from utils.metrics import cal_metrics


class Trainer(object):
    def __init__(self, args, noter):
        print('[info] Loading data')
        self.dl = get_dataloader(args)
        self.n_user, self.n_item, self.n_item_a, self.n_item_b = self.dl.dataset.get_stat()
        print('Done.\n')

        self.noter = noter
        self.device = args.device
        self.d_latent = args.d_latent
        self.n_mtc = args.n_mtc

        # model
        self.model = BERT4Rec(args).to(args.device)
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
        self.optimizer.zero_grad()
        loss = 0.
        t0 = time.time()

        # training
        self.noter.log_msg(f'\n[epoch {i_epoch:>2}]')
        for batch in tqdm(self.dl, desc='training', leave=False):
            loss_batch = self.train_batch(batch)

            n_seq = batch[0].shape[0]
            loss += (loss_batch * n_seq)

        self.noter.log_train(loss / self.n_user, time.time() - t0)

        # validating
        self.set_valid()
        ranks_a, ranks_b = [], []
        with torch.no_grad():
            for batch in tqdm(self.dl, desc='validating', leave=False):
                ranks_a_batch, ranks_b_batch = self.evaluate_batch(batch)
                ranks_a += ranks_a_batch
                ranks_b += ranks_b_batch

        return cal_metrics(ranks_a), cal_metrics(ranks_b)

    def run_test(self):
        self.set_test()
        ranks_a, ranks_b = [], []
        with torch.no_grad():
            for batch in tqdm(self.dl, desc='testing', leave=False):
                ranks_a_batch, ranks_b_batch = self.evaluate_batch(batch)
                ranks_a += ranks_a_batch
                ranks_b += ranks_b_batch

        return cal_metrics(ranks_a), cal_metrics(ranks_b)

    def train_batch(self, batch):
        seq, pos, mask_seq, gt, gt_neg, mask_gt_a, mask_gt_b = map(lambda x: x.to(self.device), batch)

        h_seq = self.model(seq, pos, mask_seq)
        loss = self.model.cal_rec_loss_dt(h_seq, gt, gt_neg, mask_gt_a, mask_gt_b)

        loss.backward()
        self.optimizer.step()
        return loss.item()

    def evaluate_batch(self, batch):
        seq, pos, mask_seq, gt, gt_mtc, mask_gt_a, mask_gt_b = map(lambda x: x.to(self.device), batch)

        h_seq = self.model(seq, pos, mask_seq)
        h_last = h_seq[:, -1]

        return self.model.cal_rank_dt(h_last, gt, gt_mtc, mask_gt_a, mask_gt_b)
