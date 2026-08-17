"""
train_tricdr.py — Phase 2: Tri-CDR training with TCA + TCL.

Generalized from Tri_CDR.py. Supports arbitrary cross-domain pairs.

Usage:
    python train_tricdr.py \
        --cross_dataset asc --dataset Sports --source Sports --target Clothing
"""

import json
import os
import sys
import io
import time
import argparse
import logging
import numpy as np
import torch
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SASRec_V12_time_final, EarlyStopping_onetower, NTXentLoss
from utils import data_partition, WarpSampler, evaluate_SASRec

# ---- Config ----
DATA_ROOT = "/nfsshare/home/liujingyan/data/CDSR/data"


def str2bool(s):
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def get_updateModel(model, path_mix, path_source_domain, path_target_domain):
    """Load pretrained SASRec weights into Tri-CDR three-tower model.

    In 'amazon_toy' branch of model.py:
      sasrec_embedding_source → used for TARGET domain items (prediction)
      sasrec_embedding_target → used for SOURCE domain items (history)
    So: source_domain checkpoint → target tower, target_domain ckpt → source tower.
    """
    pretrained_dict_mix = torch.load(path_mix, map_location='cpu')
    pretrained_dict_src_dom = torch.load(path_source_domain, map_location='cpu')
    pretrained_dict_tgt_dom = torch.load(path_target_domain, map_location='cpu')
    model_dict = model.state_dict()

    # mix ckpt → sasrec_embedding_mix
    remapped_mix = {f"sasrec_embedding_mix.{k}": v for k, v in pretrained_dict_mix.items()
                    if f"sasrec_embedding_mix.{k}" in model_dict}

    # source_domain ckpt (e.g. Sports) → sasrec_embedding_target (processes source seqs)
    remapped_target = {f"sasrec_embedding_target.{k}": v for k, v in pretrained_dict_src_dom.items()
                       if f"sasrec_embedding_target.{k}" in model_dict}

    # target_domain ckpt (e.g. Clothing) → sasrec_embedding_source (predicts target items)
    remapped_source = {f"sasrec_embedding_source.{k}": v for k, v in pretrained_dict_tgt_dom.items()
                       if f"sasrec_embedding_source.{k}" in model_dict}

    model_dict.update(remapped_mix)
    model_dict.update(remapped_source)
    model_dict.update(remapped_target)

    logging.info(f"Loaded pretrained weights: mix={len(remapped_mix)}, "
                 f"source_tower(from_target_domain)={len(remapped_source)}, "
                 f"target_tower(from_source_domain)={len(remapped_target)}")
    model.load_state_dict(model_dict)
    return model


def train(args):
    # Seed
    SEED = args.seed
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    result_path = os.path.join(args.output_dir, 'Log_TriCDR')
    os.makedirs(result_path, exist_ok=True)

    with open(os.path.join(result_path, 'args.txt'), 'w') as f:
        f.write('\n'.join([f"{k},{v}" for k, v in sorted(vars(args).items())]))

    # Determine rates
    rate_sum = args.rate_mix_source + args.rate_mix_target + args.rate_source_target
    rate_ms = args.rate_mix_source / rate_sum
    rate_mt = args.rate_mix_target / rate_sum
    rate_st = args.rate_source_target / rate_sum

    # Load data
    dataset = data_partition(args.source, args.target, args.cross_dataset,
                             args.maxlen, args.data_root)
    [user_train_mix, user_train_source, user_train_target,
     user_valid_target, user_test_target,
     seq_idx_mix, seq_idx_source,
     usernum, itemnum, interval] = dataset

    num_batch = len(user_train_source) // args.batch_size

    cc_s = sum(len(user_train_source[u]) for u in user_train_source)
    cc_t = sum(len(user_train_target[u]) for u in user_train_target)
    logging.info(f'Avg seq len: source={cc_s / len(user_train_source):.1f}, '
                 f'target={cc_t / len(user_train_source):.1f}')

    # Tri-CDR uses is_reversed=False: source ≤ interval, target > interval
    sampler = WarpSampler(
        args.source, args.target, args.cross_dataset, interval,
        user_train_mix, user_train_source, user_train_target,
        seq_idx_mix, seq_idx_source,
        usernum, itemnum, SEED,
        is_reversed=False,
        batch_size=args.batch_size, maxlen=args.maxlen, n_workers=3
    )

    # Create model
    model = SASRec_V12_time_final(usernum, itemnum, args).to(args.device)
    for name, param in model.named_parameters():
        try:
            torch.nn.init.xavier_normal_(param.data)
        except:
            pass

    # Load pretrained checkpoints.
    # In 'amazon_toy' branch of model.py, sasrec_embedding_source is used for
    # target-domain items and sasrec_embedding_target for source-domain items.
    # So: source_domain ckpt → target tower, target_domain ckpt → source tower.
    ckpt_dir = os.path.join(args.data_root, args.cross_dataset,
                            f"{args.source}_{args.target}", "Checkpoints")
    mix_ckpt = os.path.join(ckpt_dir, "SASRec_checkpoint_Mix.pt")
    source_domain_ckpt = os.path.join(ckpt_dir, f"SASRec_checkpoint_{args.source}.pt")
    target_domain_ckpt = os.path.join(ckpt_dir, f"SASRec_checkpoint_{args.target}.pt")

    # In amazon_toy: source ckpt maps to target tower, target ckpt maps to source tower
    if all(os.path.exists(p) for p in [mix_ckpt, source_domain_ckpt, target_domain_ckpt]):
        model = get_updateModel(model, mix_ckpt, source_domain_ckpt, target_domain_ckpt)
    else:
        logging.warning("Pretrained checkpoints not found, training from scratch")

    # match model.py's hardcoded fname check — source items always have lower IDs
    # in our data (1..interval), matching the 'amazon_toy' pattern
    model.fname = 'amazon_toy'

    model.train()

    bce_criterion = torch.nn.BCEWithLogitsLoss()
    cl_criterion = NTXentLoss(temperature=args.info_NCE_temperature)
    triplet_criterion = torch.nn.TripletMarginLoss(
        margin=args.triplet_margin, p=2.0, reduction='mean'
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.lr_decay_rate)
    early_stopping = EarlyStopping_onetower(args.patience, version='TriCDR', verbose=True)

    t0 = time.time()

    for epoch in range(1, args.num_epochs + 1):
        t1 = time.time()
        loss_rec_val = loss_cl_val = loss_trip_val = loss_val = 0.0
        dist_ms = dist_mt = dist_st = 0.0

        if args.inference_only:
            break

        for step in range(num_batch):
            u, seq_mix, seq_source, seq_target, pos_target, neg_target, \
                mix_idx, source_idx = sampler.next_batch()

            u, seq_mix, seq_source, seq_target = np.array(u), np.array(seq_mix), \
                np.array(seq_source), np.array(seq_target)
            pos_target, neg_target = np.array(pos_target), np.array(neg_target)
            mix_idx, source_idx = np.array(mix_idx), np.array(source_idx)

            mix_feats, source_feats, target_feats, pos_logits, neg_logits = model(
                u, seq_mix, seq_source, seq_target,
                pos_target, neg_target, mix_idx, source_idx
            )

            pos_labels = torch.ones(pos_logits.shape, device=args.device)
            neg_labels = torch.zeros(neg_logits.shape, device=args.device)

            optimizer.zero_grad()
            indices = np.where(pos_target != 0)
            loss_rec = bce_criterion(pos_logits[indices], pos_labels[indices])
            loss_rec += bce_criterion(neg_logits[indices], neg_labels[indices])

            cl_mix_source = cl_criterion(mix_feats, source_feats)
            cl_mix_target = cl_criterion(mix_feats, target_feats)
            cl_source_target = cl_criterion(source_feats, target_feats)
            loss_cl = cl_mix_source * rate_ms + cl_mix_target * rate_mt + cl_source_target * rate_st

            loss_triplet = triplet_criterion(source_feats, mix_feats, target_feats)

            loss = loss_rec + loss_cl * args.cl_weight + loss_triplet * args.triplet_weight
            loss_rec_val += loss_rec.item()
            loss_cl_val += loss_cl.item() * args.cl_weight
            loss_trip_val += loss_triplet.item() * args.triplet_weight
            loss_val += loss.item()

            loss.backward()
            optimizer.step()

        scheduler.step()

        logging.info(f"Epoch {epoch:3d}: rec={loss_rec_val / num_batch:.4f}, "
                     f"cl={loss_cl_val / num_batch:.4f}, trip={loss_trip_val / num_batch:.4f}, "
                     f"total={loss_val / num_batch:.4f}, time={time.time() - t1:.1f}s")

        # Evaluate
        model.eval()
        t_test = evaluate_SASRec(model, dataset, args)
        logging.info(f'  Test: NDCG@5={t_test[1]:.4f} NDCG@10={t_test[2]:.4f} '
                     f'HR@5={t_test[6]:.4f} HR@10={t_test[7]:.4f} AUC={t_test[10]:.4f}')

        model.train()
        early_stopping(epoch, model, result_path, t_test)
        if early_stopping.early_stop:
            logging.info(f"Early stopping at epoch {epoch}, "
                         f"best NDCG@10={early_stopping.best_performance[2]:.4f}")
            break

    sampler.close()
    T = time.time() - t0
    logging.info(f"Training complete. Total time: {T:.1f}s")

    # Save final results as JSON for easy comparison
    best = early_stopping.best_performance
    if best is not None:
        results = {
            "pair": args.cross_dataset,
            "direction": f"{args.source}→{args.target}",
            "NDCG@5": round(float(best[1]), 6),
            "NDCG@10": round(float(best[2]), 6),
            "HR@5": round(float(best[6]), 6),
            "HR@10": round(float(best[7]), 6),
            "AUC": round(float(best[10]), 6),
            "best_epoch": early_stopping.save_epoch,
        }
        results_path = os.path.join(result_path, "results.json")
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logging.info(f"Best results saved to: {results_path}")
        logging.info(f"Best: NDCG@5={results['NDCG@5']:.4f} NDCG@10={results['NDCG@10']:.4f} "
                     f"HR@5={results['HR@5']:.4f} HR@10={results['HR@10']:.4f}")


def main():
    parser = argparse.ArgumentParser(description='Tri-CDR Training')
    # Required args
    parser.add_argument('--cross_dataset', required=True, help='Pair name (asc/ape/dbm/ghk)')
    parser.add_argument('--source', required=True, help='Source domain name')
    parser.add_argument('--target', required=True, help='Target domain name')
    parser.add_argument('--dataset', required=True, help='Name of dataset for prediction target')

    # Training params
    parser.add_argument('--batch_size', default=120, type=int)
    parser.add_argument('--lr', default=0.001, type=float)
    parser.add_argument('--maxlen', default=200, type=int)
    parser.add_argument('--hidden_units', default=64, type=int)
    parser.add_argument('--num_blocks', default=2, type=int)
    parser.add_argument('--num_epochs', default=1000, type=int)
    parser.add_argument('--num_heads', default=1, type=int)
    parser.add_argument('--dropout_rate', default=0.2, type=float)
    parser.add_argument('--l2_emb', default=0.0, type=float)
    parser.add_argument('--device', default='cuda', type=str)
    parser.add_argument('--inference_only', default=False, type=str2bool)
    parser.add_argument('--num_samples', default=100, type=int)
    parser.add_argument('--decay', default=4, type=int)
    parser.add_argument('--lr_decay_rate', default=0.99, type=float)
    parser.add_argument('--temperature', default=5.0, type=float)
    parser.add_argument('--seed', default=5, type=int)
    parser.add_argument('--patience', default=10, type=int)
    parser.add_argument('--info_NCE_temperature', default=0.1, type=float)
    parser.add_argument('--rate_mix_source', default=1.0, type=float)
    parser.add_argument('--rate_mix_target', default=1.0, type=float)
    parser.add_argument('--rate_source_target', default=1.0, type=float)
    parser.add_argument('--cl_weight', default=1.0, type=float)
    parser.add_argument('--triplet_weight', default=1.0, type=float)
    parser.add_argument('--triplet_margin', default=1.0, type=float)
    parser.add_argument('--lrscheduler', default='ExponentialLR', type=str)

    # Path args
    parser.add_argument('--data_root', type=str, default=DATA_ROOT)
    parser.add_argument('--output_dir', type=str, default=None)

    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_root, args.cross_dataset,
                                       f"{args.source}_{args.target}")

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Tri-CDR: {args.source} → {args.target} (pair={args.cross_dataset})")
    logging.info(f"Data: {args.data_root}")
    logging.info(f"Output: {args.output_dir}")

    train(args)


if __name__ == '__main__':
    main()
