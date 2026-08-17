import argparse
import os
import random
from os.path import join

import numpy as np
import torch

from models.MERIT import MERIT
from models.data.dataloader import get_dataloader


def main() -> None:
    parser = argparse.ArgumentParser(description="MERIT smoke test")
    parser.add_argument("--data", type=str, default="ape")
    parser.add_argument("--path_data_root", type=str, default=None)
    parser.add_argument("--cuda", type=str, default="0")
    parser.add_argument("--len_max", type=int, default=50)
    parser.add_argument("--bs", type=int, default=2)
    parser.add_argument("--eval_bs", type=int, default=2)
    parser.add_argument("--n_worker", type=int, default=0)
    parser.add_argument("--d_embed", type=int, default=16)
    parser.add_argument("--n_head", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--temp", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=3407)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    args.raw = False
    args.n_neg = 4
    args.n_mtc = 999
    args.eval_mode = "full"
    args.len_trim = args.len_max - 3
    args.bse = args.eval_bs

    args.path_root = os.getcwd()
    if args.path_data_root is None:
        args.path_data_root = join(os.path.dirname(args.path_root), "data")
    args.path_data = join(args.path_data_root, args.data)
    args.f_data = join(args.path_data, f"{args.data}_{args.len_max}_seq.pkl")
    args.f_raw = join(args.path_data, f"{args.data}_{args.len_max}_preprocessed.txt")

    if args.cuda == "cpu":
        args.device = torch.device("cpu")
    else:
        args.device = torch.device(f"cuda:{args.cuda}")

    print(f"[smoke] cwd={args.path_root}")
    print(f"[smoke] data={args.f_data}")
    print(f"[smoke] torch={torch.__version__}, cuda_available={torch.cuda.is_available()}")

    train_loader, val_loader, _ = get_dataloader(args)
    model = MERIT(args).to(args.device).eval()

    train_batch = next(iter(train_loader))
    print("[smoke] train shapes:", [tuple(x.shape) for x in train_batch])

    batch = [x.to(args.device) for x in next(iter(val_loader))]
    seq_m, idx_last_a, idx_last_b, gt, gt_mtc, gt_mtc_mask = batch
    with torch.no_grad():
        h_m, h_a, h_b = model(seq_m, idx_last_a, idx_last_b)
        ranks, mask_a, mask_b = model.cal_rank(h_m, h_a, h_b, gt, gt_mtc, gt_mtc_mask)

    print("[smoke] eval ranks:", ranks.detach().cpu().tolist())
    print("[smoke] domain masks:", mask_a.squeeze(-1).detach().cpu().tolist(),
          mask_b.squeeze(-1).detach().cpu().tolist())
    print("[smoke] ok")


if __name__ == "__main__":
    main()
