import argparse
import os
import random
from os.path import join
import numpy as np
import torch
from tqdm import tqdm

from models.MERIT import MERIT
from models.data.evaluation import cal_metrics
from models.data.dataloader import get_dataloader


def main() -> None:
    parser = argparse.ArgumentParser(description='MERIT-Evaluation')
    parser.add_argument('--ckpt', type=str, required=True, help='path to model checkpoint (.pth)')
    parser.add_argument('--data', type=str, default='abe', help='afk: Food-Kitchen'
                                                                'abe: Beauty-Electronics'
                                                                'amb: Movie-Book')
    parser.add_argument('--path_data_root', type=str, default=None,
                        help='root directory that contains dataset folders; defaults to ../data if present')
    parser.add_argument('--len_max', type=int, default=50, help='# of interactions allowed to input')
    parser.add_argument('--eval_mode', type=str, default='full', choices=['full', 'sampled'],
                        help='full ranks against every unseen in-domain item; sampled uses n_mtc negatives')
    parser.add_argument('--n_mtc', type=int, default=999, help='# negative metric samples when --eval_mode sampled')
    parser.add_argument('--eval_bs', type=int, default=None, help='evaluation batch size')

    # Model (must match training config)
    parser.add_argument('--d_embed', type=int, default=256, help='dimension of latent representation')
    parser.add_argument('--n_attn', type=int, default=1, help='# layer of TransformerEncoderLayer stack')
    parser.add_argument('--n_head', type=int, default=2, help='# multi-head for self-attention')
    parser.add_argument('--dropout', type=float, default=0.5, help='dropout rate')
    parser.add_argument('--temp', type=float, default=0.75, help='temperature for InfoNCE')

    # System
    parser.add_argument('--cuda', type=str, default='0', help='running device')
    parser.add_argument('--seed', type=int, default=3407, help='random seeding')
    parser.add_argument('--bs', type=int, default=256, help='batch size (only for dataloader init)')
    parser.add_argument('--n_worker', type=int, default=0, help='# dataloader worker')

    args = parser.parse_args()

    # seeding
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    os.environ['PYTHONHASHSEED'] = str(args.seed)

    if args.cuda == 'cpu':
        args.device = torch.device('cpu')
    else:
        args.device = torch.device(f'cuda:{args.cuda}')

    args.raw = False
    args.n_neg = 128
    args.len_trim = args.len_max - 3  # leave-one-out
    args.bse = args.eval_bs or (8 if args.eval_mode == 'full' else args.bs * 4)

    # paths
    args.path_root = os.getcwd()
    if args.path_data_root is None:
        path_data_root = join(os.path.dirname(args.path_root), 'data')
        if not os.path.exists(path_data_root):
            path_data_root = join(args.path_root, 'data')
        args.path_data_root = path_data_root
    args.path_data = join(args.path_data_root, args.data)
    args.f_raw = join(args.path_data, args.data + f'_{args.len_max}_preprocessed.txt')
    args.f_data = join(args.path_data, args.data + f'_{args.len_max}_seq.pkl')

    print(f'[info] Loading data from {args.path_data}')
    _, _, test_loader = get_dataloader(args)
    print('Done.\n')

    # build model
    print(f'[info] Building model (d_embed={args.d_embed}, n_head={args.n_head})')
    model = MERIT(args).to(args.device)

    # load checkpoint
    print(f'[info] Loading checkpoint from {args.ckpt}')
    state_dict = torch.load(args.ckpt, map_location=args.device)
    # handle both raw state_dict and wrapped dict (e.g. {"model": ..., "epoch": ...})
    if isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']
    model.load_state_dict(state_dict)
    print('Checkpoint loaded.\n')

    # evaluate
    model.eval()
    ranks_f2a, ranks_f2b = [], []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc='evaluating'):
            seq_m, idx_last_a, idx_last_b, gt, gt_mtc, gt_mtc_mask = map(lambda x: x.to(args.device), batch)

            h_m, h_a, h_b = model(seq_m, idx_last_a, idx_last_b)
            ranks, mask_gt_a, mask_gt_b = model.cal_rank(h_m, h_a, h_b, gt, gt_mtc, gt_mtc_mask)
            ranks_a = ranks[mask_gt_a.squeeze(-1) == 1].tolist()
            ranks_b = ranks[mask_gt_b.squeeze(-1) == 1].tolist()

            ranks_f2a += ranks_a
            ranks_f2b += ranks_b

    res_a = cal_metrics(ranks_f2a)
    res_b = cal_metrics(ranks_f2b)

    # print results
    print(f'\n{"=" * 60}')
    print(f'[Evaluation Result] data={args.data}, ckpt={args.ckpt}')
    print(f'{"=" * 60}')
    print(f'|                     A                      |                     B                      |')
    print(f'|  hr5   |  hr10  | ndcg5  | ndcg10 |  mrr   |  hr5   |  hr10  | ndcg5  | ndcg10 |  mrr   |')
    print(f'| {res_a[0]:.4f} | {res_a[1]:.4f} | {res_a[2]:.4f} | {res_a[3]:.4f} | {res_a[4]:.4f} | {res_b[0]:.4f} | {res_b[1]:.4f} | {res_b[2]:.4f} | {res_b[3]:.4f} | {res_b[4]:.4f} |')
    print(f'{"=" * 60}\n')


if __name__ == '__main__':
    main()
