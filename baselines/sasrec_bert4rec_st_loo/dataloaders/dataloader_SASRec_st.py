from os.path import join
import numpy as np
import json
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader


class CDSRDataset(Dataset):
    def __init__(self, args):
        self.mode = 'train'

        self.path_data = args.path_data
        self.f_raw = args.f_raw
        self.device = args.device

        self.idx_pad = 0
        self.n_mtc = args.n_mtc
        self.eval_mode = getattr(args, 'eval_mode', 'sampled')
        self.len_max = args.len_max
        self.len_trim = args.len_trim

        # load data
        if args.raw:
            print('Reading raw data...')
            data_seq = self.read_data()

            self.data_tr, self.data_val, self.data_te = self.serialize_data(data_seq)
            print('Saving serialized seqs...')
            with open(args.f_data, 'wb') as f:
                pickle.dump((self.data_tr, self.data_val, self.data_te, self.n_user, self.n_item), f)

        else:
            print('Loading serialized seqs...')
            with open(args.f_data, 'rb') as f:
                self.data_tr, self.data_val, self.data_te, self.n_user, self.n_item = pickle.load(f)

        args.n_user = self.length = self.n_user
        args.n_item = self.n_item

        self.idx_all = np.arange(1, self.n_item + 1)

    def read_data(self):
        """ read preprocessed file """
        with open(join(self.path_data, 'map_user.txt'), 'r') as f:
            self.n_user = len(json.load(f))

        with open(join(self.path_data, 'map_item.txt'), 'r') as f:
            self.n_item = len(json.load(f))

        data_seq = []
        with open(join(self.path_data, self.f_raw), 'r', encoding='utf-8') as f:
            for line in f:
                seq = []
                line = line.strip().split(' ')
                for ui in line[1:][-self.len_max:]:
                    ui = ui.split('|')
                    seq.append(int(ui[0]))

                data_seq.append(np.array(seq))

        return data_seq

    def serialize_data(self, data_seq):
        """ serialize data """
        serialized_tr = []
        serialized_val = []
        serialized_te = []

        for seq in tqdm(data_seq, desc='processing', leave=False):
            serialized_tr.append(self.process_train(seq))
            serialized_val.append(self.process_valid(seq))
            serialized_te.append(self.process_test(seq))

        return serialized_tr, serialized_val, serialized_te

    def get_stat(self):
        """ return counts of users and items'"""
        return self.n_user, self.n_item

    def set_train(self):
        self.mode = 'train'

    def set_valid(self):
        self.mode = 'valid'

    def set_test(self):
        self.mode = 'test'

    def trim_seq(self, seq):
        """ pad sequences to required length """
        return np.concatenate((np.zeros(max(0, self.len_trim - len(seq)), dtype=np.int32), seq))[-self.len_trim:]

    @staticmethod
    def get_pos_idx(mask):
        """ get position indices """
        pos = np.cumsum(mask) * mask
        return pos

    def process_train(self, seq_raw):
        seq = self.trim_seq(np.array(seq_raw[:-3]))
        gt = self.trim_seq(np.array(seq_raw[1:-2]))

        mask_seq = (seq != 0).astype(int)
        pos = self.get_pos_idx(mask_seq)
        mask_seq = np.expand_dims(mask_seq, -1)

        return seq, pos, mask_seq, gt

    def process_valid(self, seq_raw):
        seq = self.trim_seq(np.array(seq_raw[:-2]))
        gt = np.expand_dims(seq_raw[-2], 0)

        mask_seq = (seq != 0).astype(int)
        pos = self.get_pos_idx(mask_seq)
        mask_seq = np.expand_dims(mask_seq, -1)

        return seq, pos, mask_seq, gt, seq_raw[:-1]

    def process_test(self, seq_raw):
        seq = self.trim_seq(np.array(seq_raw[:-1]))
        gt = np.expand_dims(seq_raw[-1], 0)

        mask_seq = (seq != 0).astype(int)
        pos = self.get_pos_idx(mask_seq)
        mask_seq = np.expand_dims(mask_seq, -1)

        return seq, pos, mask_seq, gt, seq_raw

    def get_spe_neg(self, gt, n, seq):
        """ get random negative samples from observed items in specific domain """
        if gt == 0:
            return np.zeros(n, dtype=np.int32)

        else:
            gt_neg = np.random.choice(self.idx_all, n + len(seq), replace=False)
            gt_neg = gt_neg[~np.isin(gt_neg, seq[seq > 0])][:n]
            return gt_neg

    def get_full_eval_mask(self, seq):
        """ return full-ranking negative candidate mask; observed items are excluded """
        mask = np.ones(self.n_item + 1, dtype=np.int32)
        mask[0] = 0
        seq = np.asarray(seq)
        mask[seq[seq > 0]] = 0
        return mask

    def get_item_eval(self, index, data):
        seq, pos, mask_seq, gt, seq_raw = data[index]
        if self.eval_mode == 'full':
            gt_mtc = self.get_full_eval_mask(seq_raw)
        else:
            gt_mtc = self.get_spe_neg(gt, self.n_mtc, seq_raw)
        return seq, pos, mask_seq, gt, gt_mtc

    def __getitem__(self, index):
        if self.mode == 'train':
            data = self.data_tr[index]

        elif self.mode == 'valid':
            data = self.get_item_eval(index, self.data_val)

        else:
            # assert self.mode == 'test'
            data = self.get_item_eval(index, self.data_te)
        return tuple(map(lambda x: torch.LongTensor(x), data))

    def __len__(self):
        return self.length


def get_dataloader(args):
    return DataLoader(CDSRDataset(args), batch_size=args.bs, shuffle=True, num_workers=args.n_worker,
                      pin_memory=True)
