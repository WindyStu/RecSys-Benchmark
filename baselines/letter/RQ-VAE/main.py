import argparse
import random
import torch
import numpy as np
from time import time
import logging
import datetime
import wandb
from torch.utils.data import DataLoader

from datasets import EmbDataset
from models.rqvae import RQVAE
from trainer import  Trainer
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"

def str2bool(value):
    if isinstance(value, bool):
        return value
    value = value.lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError("Boolean value expected.")

def infer_dataset_name(data_path):
    base = os.path.basename(data_path)
    return base.split(".")[0] if base else "dataset"

def setup_key_logger(log_file, data_path, stage):
    if log_file is None:
        dataset = infer_dataset_name(data_path)
        run_ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        module_dir = os.path.dirname(os.path.abspath(__file__))
        log_file = os.path.join(module_dir, "log", f"{dataset}_{stage}_{run_ts}.log")
    os.makedirs(os.path.dirname(log_file) or ".", exist_ok=True)
    file_handler = logging.FileHandler(log_file, mode="a")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
    logging.basicConfig(level=logging.INFO, handlers=[file_handler], force=True)
    return logging.getLogger(), log_file

def parse_args():
    parser = argparse.ArgumentParser(description="RQ-VAE")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=20000, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=1024, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=2000, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument("--data_path", type=str, default="../data", help="Input data path.")

    parser.add_argument('--weight_decay', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=str2bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0, 0.003], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")

    parser.add_argument("--device", type=str, default="cuda:4", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument('--alpha', type=float, default=0.1, help='cf loss weight')
    parser.add_argument('--beta', type=float, default=0.1, help='diversity loss weight')
    parser.add_argument('--no_diversity', action='store_true', default=False, help='disable diversity regularization in VQ loss')
    parser.add_argument('--n_clusters', type=int, default=10, help='n_clusters')
    parser.add_argument('--sample_strategy', type=str, default="all", help='sample_strategy')
    parser.add_argument('--cf_emb', type=str, default="./RQ-VAE/ckpt/Instruments-32d-sasrec.pt", help='cf emb')
    parser.add_argument('--no_cf', action='store_true', default=False, help='disable collaborative regularization')
    parser.add_argument('--patience', type=int, default=10, help='early stop patience by eval rounds, <=0 disables early stop')
    parser.add_argument('--early_stop_min_delta', type=float, default=0.0, help='minimum collision-rate improvement for early stop')
   
    parser.add_argument('--layers', type=int, nargs='+', default=[2048,1024,512,256,128,64], help='hidden sizes of every layer')

    parser.add_argument("--ckpt_dir", type=str, default="../checkpoint", help="output directory for model")
    parser.add_argument("--log_file", type=str, default=None, help="key log file path")

    return parser.parse_args()


if __name__ == '__main__':
    """fix the random seed"""
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()

    print(args)
    logger, key_log_file = setup_key_logger(args.log_file, args.data_path, "rqvae")
    logger.info("key log file: %s", key_log_file)
    logger.info("RQ-VAE args: %s", vars(args))
    if args.no_cf:
        args.alpha = 0.0
        cf_emb = None
        print("Collaborative regularization disabled: no CF embedding will be loaded.")
        logger.info("Collaborative regularization disabled.")
    else:
        cf_emb = torch.load(args.cf_emb).squeeze().detach().numpy()
        logger.info("Loaded CF embedding: %s, shape=%s", args.cf_emb, cf_emb.shape)

    if args.no_diversity:
        print("Diversity regularization disabled: VQ loss will not include diversity loss.")
        logger.info("Diversity regularization disabled.")

    """build dataset"""
    data = EmbDataset(args.data_path)
    logger.info("Loaded item embedding data: path=%s, items=%d, dim=%d", args.data_path, len(data), data.dim)
    model = RQVAE(in_dim=data.dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  beta = args.beta,
                  use_diversity = not args.no_diversity,
                  alpha = args.alpha,
                  n_clusters= args.n_clusters,
                  sample_strategy =args.sample_strategy,
                  cf_embedding = cf_emb
                  )
    print("model built:", model.__class__.__name__)
    logger.info("Model built: %s", model.__class__.__name__)
    data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)

    trainer = Trainer(args,model)
    best_loss, best_collision_rate = trainer.fit(data_loader)

    print("Best Loss",best_loss)
    print("Best Collision Rate", best_collision_rate)
