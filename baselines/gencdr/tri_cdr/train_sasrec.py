"""
train_sasrec.py — Phase 1: SASRec single-domain pretraining for Tri-CDR.

Trains a standalone SASRec model on source/target/mix sequences.
Outputs checkpoint compatible with Tri-CDR's get_updateModel().

Usage:
    python train_sasrec.py --pair asc --domain Sports --source Sports --target Clothing
    python train_sasrec.py --pair asc --domain Mix --source Sports --target Clothing
"""

import os
import sys
import json
import argparse
import pickle
import logging
import time
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import SASRec_Embedding

# ---- Config ----
DATA_ROOT = "/nfsshare/home/liujingyan/data/CDSR/data"


def load_sequences(data_root, pair_name, direction_dir, domain_name):
    """Load a single domain's user sequences from pkl."""
    if domain_name.lower() == 'mix':
        path = os.path.join(data_root, pair_name, direction_dir, "mix_log_file_final.pkl")
    else:
        path = os.path.join(data_root, pair_name, direction_dir,
                            f"{domain_name}_log_file_final.pkl")

    with open(path, 'rb') as f:
        data = pickle.load(f)

    # Load metadata to get total itemnum (used for embedding size)
    meta_path = os.path.join(data_root, pair_name, direction_dir, "meta.json")
    with open(meta_path, 'r') as f:
        meta = json.load(f)

    logging.info(f"Loaded {domain_name} sequences: {len(data)} users from {path}")
    logging.info(f"  meta: itemnum={meta['itemnum']}, interval={meta['interval']}")
    return data, meta


def build_train_data(user_seqs, maxlen):
    """Build training samples from user sequences (SASRec style).

    For each user, for each position i >= 1 in the sequence:
        input: [item_{i-maxlen}, ..., item_{i-1}] padded to maxlen
        pos:   item_i
        neg:   random item ≠ item_i
    """
    all_users = list(user_seqs.keys())
    item_set = set()
    for seq in user_seqs.values():
        item_set.update(seq)
    item_list = sorted(list(item_set))

    return user_seqs, all_users, item_set, item_list


def random_neg(l, r, exclude):
    t = np.random.randint(l, r)
    while t in exclude:
        t = np.random.randint(l, r)
    return t


def evaluate(model, user_seqs, all_users, item_list, maxlen, device, batch_size=128):
    """Evaluate NDCG@10 and HR@10 on validation set (leave-one-out)."""
    model.eval()
    ndcg_10 = 0.0
    hr_10 = 0.0
    ndcg_5 = 0.0
    hr_5 = 0.0
    valid_users = 0

    with torch.no_grad():
        for u in all_users:
            seq = user_seqs[u]
            if len(seq) < 3:
                continue

            # Use the validation item for checkpoint selection; keep the last
            # item untouched for final test evaluation.
            target = seq[-2]
            history = seq[:-2]

            # Build input sequence (padded)
            input_seq = np.zeros([maxlen], dtype=np.int32)
            idx = maxlen - 1
            for item in reversed(history):
                input_seq[idx] = item
                idx -= 1
                if idx == -1:
                    break

            # Sample negative items + true item
            exclude = set(seq)
            neg_samples = min(100, len(item_list))
            sample_pool = [i for i in item_list if i not in exclude]
            if len(sample_pool) >= neg_samples:
                candidates = list(np.random.choice(sample_pool, neg_samples, replace=False))
            else:
                candidates = sample_pool
            candidates = [target] + candidates

            # log2feats accepts numpy, handles device internally
            log_feats = model.log2feats(np.array([input_seq]))  # (1, maxlen, dim)
            user_feat = log_feats[:, -1, :]  # last position
            user_feat = nn.functional.normalize(user_feat, p=2, dim=1)

            item_embs = model.item_emb(torch.LongTensor(candidates).to(device))
            item_embs = nn.functional.normalize(item_embs, p=2, dim=1)

            scores = item_embs.matmul(user_feat.T).squeeze()  # (candidates,)
            _, rank_idx = scores.sort(descending=True)
            rank = (rank_idx == 0).nonzero(as_tuple=True)[0].item()

            valid_users += 1
            if rank < 5:
                ndcg_5 += 1.0 / np.log2(rank + 2)
                hr_5 += 1
            if rank < 10:
                ndcg_10 += 1.0 / np.log2(rank + 2)
                hr_10 += 1

    return ndcg_5 / valid_users, ndcg_10 / valid_users, hr_5 / valid_users, hr_10 / valid_users


class Args:
    """Minimal args object matching Tri-CDR expectations."""
    def __init__(self):
        self.hidden_units = 64
        self.num_blocks = 2
        self.num_heads = 1
        self.dropout_rate = 0.2
        self.maxlen = 80
        self.device = 'cuda'
        self.l2_emb = 0.0


def train(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_args = Args()

    # Determine direction directory
    direction_dir = f"{args.source}_{args.target}"

    # Load data and metadata
    user_seqs, meta = load_sequences(args.data_root, args.pair, direction_dir, args.domain)
    user_seqs_dict, all_users, item_set, item_list = build_train_data(user_seqs, model_args.maxlen)

    # Use total itemnum (mix) for embedding size — matches Tri-CDR model
    itemnum = meta['itemnum']
    usernum = max(user_seqs_dict.keys())

    logging.info(f"Users: {usernum}, Items (max id): {itemnum}, Unique items: {len(item_set)}")
    logging.info(f"Average seq length: {np.mean([len(v) for v in user_seqs_dict.values()]):.1f}")

    # Create model
    model = SASRec_Embedding(itemnum, model_args).to(device)
    model.train()

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.98))
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)

    bce_criterion = nn.BCEWithLogitsLoss()

    # Training loop
    best_ndcg10 = 0.0
    patience = 10
    patience_counter = 0
    best_epoch = 0

    random_min = 1
    random_max = itemnum + 1

    epoch_times = []

    for epoch in range(1, 1001):
        t0 = time.time()
        model.train()
        total_loss = 0.0
        num_batches = 0

        for u in all_users:
            seq = user_seqs_dict[u]
            if len(seq) < 3:  # need at least train+valid+test
                continue

            # For SASRec: use all items except last as history, predict each next
            train_seq = seq[:-2]  # leave valid and test out
            test_seq = seq[-2:]  # last two for validation

            for i in range(1, len(train_seq)):
                # Build input up to position i
                input_len = min(i, model_args.maxlen)
                input_seq = np.zeros([model_args.maxlen], dtype=np.int32)
                idx = model_args.maxlen - 1
                for j in range(i - input_len, i):
                    input_seq[idx] = train_seq[j]
                    idx -= 1
                    if idx == -1:
                        break

                pos_item = train_seq[i]
                neg_item = random_neg(random_min, random_max, set(train_seq))

                # log2feats accepts numpy (batch, maxlen), handles device internally
                log_feats = model.log2feats(np.array([input_seq]))  # (1, maxlen, dim)
                user_feat = log_feats[:, -1, :]  # (1, dim)

                pos_emb = model.item_emb(torch.LongTensor([pos_item]).to(device))
                neg_emb = model.item_emb(torch.LongTensor([neg_item]).to(device))

                pos_logit = (user_feat * pos_emb).sum(dim=-1)
                neg_logit = (user_feat * neg_emb).sum(dim=-1)

                pos_label = torch.ones_like(pos_logit)
                neg_label = torch.zeros_like(neg_logit)

                loss = bce_criterion(pos_logit, pos_label) + bce_criterion(neg_logit, neg_label)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                num_batches += 1

        scheduler.step()
        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        avg_loss = total_loss / max(num_batches, 1)

        # Evaluate every epoch on validation set (using last-1 as target)
        ndcg5, ndcg10, hr5, hr10 = evaluate(
            model, user_seqs_dict, all_users, item_list, model_args.maxlen, device
        )

        logging.info(f"Epoch {epoch:3d}: loss={avg_loss:.4f}, NDCG@5={ndcg5:.4f}, "
                     f"NDCG@10={ndcg10:.4f}, HR@5={hr5:.4f}, HR@10={hr10:.4f}, "
                     f"time={epoch_time:.1f}s")

        if ndcg10 > best_ndcg10:
            best_ndcg10 = ndcg10
            patience_counter = 0
            best_epoch = epoch
            # Save checkpoint
            ckpt_dir = os.path.join(args.output_dir or ".", "Checkpoints")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"SASRec_checkpoint_{args.domain}.pt")
            torch.save(model.state_dict(), ckpt_path)
            logging.info(f"  → Saved best checkpoint: {ckpt_path}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logging.info(f"Early stopping at epoch {epoch}, best was epoch {best_epoch} "
                             f"with NDCG@10={best_ndcg10:.4f}")
                break

    logging.info(f"Training complete. Best NDCG@10={best_ndcg10:.4f} at epoch {best_epoch}")


def main():
    parser = argparse.ArgumentParser(description="SASRec single-domain pretraining for Tri-CDR")
    parser.add_argument("--pair", type=str, required=True, help="Cross-domain pair (asc/ape/dbm/ghk)")
    parser.add_argument("--domain", type=str, required=True,
                        help="Domain name or 'Mix'")
    parser.add_argument("--source", type=str, required=True, help="Source domain name")
    parser.add_argument("--target", type=str, required=True, help="Target domain name")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for checkpoint (default: data_root/pair/source_target)")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(args.data_root, args.pair,
                                       f"{args.source}_{args.target}")

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')
    logging.info(f"Training SASRec: pair={args.pair} domain={args.domain}")
    logging.info(f"Direction: {args.source} → {args.target}")
    logging.info(f"Output: {args.output_dir}")

    train(args)


if __name__ == "__main__":
    main()
