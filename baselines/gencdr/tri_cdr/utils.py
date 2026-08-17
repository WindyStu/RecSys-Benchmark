"""
utils.py — Generalized data loading and evaluation for Tri-CDR.

Adapted from Tri-CDR utils.py with parametrized paths and
removal of hardcoded amazon_toy/amazon_game logic.
"""

import sys
import copy
import json
import torch
import random
import numpy as np
import os
import pickle
import logging
from collections import defaultdict
from multiprocessing import Process, Queue


def roc_auc_score(y_true, y_score):
    """Manual AUC implementation — avoids sklearn dependency."""
    desc_idx = np.argsort(y_score)[::-1]
    y_true = np.array(y_true)[desc_idx]
    n_pos = y_true.sum()
    n_neg = len(y_true) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5
    tpr = np.cumsum(y_true) / n_pos
    fpr = np.cumsum(1 - y_true) / n_neg
    # Trapezoidal integration
    return np.sum((tpr[1:] + tpr[:-1]) * (fpr[1:] - fpr[:-1]) / 2.0)


def random_neq(l, r, s):
    t = np.random.randint(l, r)
    while t in s:
        t = np.random.randint(l, r)
    return t


def sample_function(source_name, target_name, cross_dataset, interval,
                    user_train_mix, user_train_source, user_train_target,
                    seq_idx_mix, seq_idx_source,
                    usernum, itemnum, batch_size, maxlen,
                    result_queue, seed, is_reversed=False):
    """Multi-process batch sampler for Tri-CDR training.

    If is_reversed: source has higher IDs (source→target direction reversed).
    """
    # Determine item ranges based on direction
    if not is_reversed:
        # source items: 1..interval, target items: interval+1..itemnum
        source_min, source_max = 1, interval + 1
        target_min, target_max = interval + 1, itemnum + 1
    else:
        # reversed: target items 1..interval, source items interval+1..itemnum
        target_min, target_max = 1, interval + 1
        source_min, source_max = interval + 1, itemnum + 1

    # For negative sampling: we sample from the target domain
    random_min, random_max = target_min, target_max

    def sample():
        user = np.random.randint(1, usernum + 1)
        while (user not in user_train_mix or
               len(user_train_mix.get(user, [])) <= 1 or
               len(user_train_source.get(user, [])) <= 1 or
               len(user_train_target.get(user, [])) <= 1):
            user = np.random.randint(1, usernum + 1)

        seq_mix = np.zeros([maxlen], dtype=np.int32)
        seq_source = np.zeros([maxlen], dtype=np.int32)
        seq_target = np.zeros([maxlen], dtype=np.int32)
        pos_target = np.zeros([maxlen], dtype=np.int32)
        neg_target = np.zeros([maxlen], dtype=np.int32)
        mix_idx_for_target = np.zeros([maxlen], dtype=np.int32)
        source_idx_for_target = np.zeros([maxlen], dtype=np.int32)

        target_seq = user_train_target[user]
        if len(target_seq) < 1:
            return (user, seq_mix, seq_source, seq_target, pos_target, neg_target,
                    mix_idx_for_target, source_idx_for_target)

        nxt_target = target_seq[-1]

        idx_mix = maxlen - 1
        idx_source = maxlen - 1
        idx_target = maxlen - 1

        # Fill mix sequence (reversed order)
        ts_target = set(target_seq)
        for item in reversed(user_train_mix[user]):
            seq_mix[idx_mix] = item
            idx_mix -= 1
            if idx_mix == -1:
                break

        # Fill source sequence
        for item in reversed(user_train_source[user]):
            seq_source[idx_source] = item
            idx_source -= 1
            if idx_source == -1:
                break

        # Fill target sequence with positions and negatives
        target_history = target_seq[:-1]
        for item in reversed(target_history):
            seq_target[idx_target] = item
            pos_target[idx_target] = nxt_target

            # Sequence index alignment
            if seq_idx_mix is not None:
                mix_idx_val = seq_idx_mix.get(user, [0] * len(target_history))
                idx_in_history = len(target_history) - 1 - target_history[::-1].index(item) if item in target_history else 0
                try:
                    mix_val = mix_idx_val[min(idx_in_history, len(mix_idx_val) - 1)]
                except:
                    mix_val = 0
                if mix_val < -maxlen:
                    mix_idx_for_target[idx_target] = 0
                else:
                    mix_idx_for_target[idx_target] = mix_val + maxlen
            else:
                mix_idx_for_target[idx_target] = maxlen

            if seq_idx_source is not None:
                source_idx_val = seq_idx_source.get(user, [0] * len(target_history))
                try:
                    source_val = source_idx_val[min(idx_in_history, len(source_idx_val) - 1)]
                except:
                    source_val = 0
                if source_val < -maxlen:
                    source_idx_for_target[idx_target] = 0
                else:
                    source_idx_for_target[idx_target] = source_val + maxlen
            else:
                source_idx_for_target[idx_target] = maxlen

            if nxt_target != 0:
                neg_target[idx_target] = random_neq(random_min, random_max, ts_target)
            nxt_target = item
            idx_target -= 1
            if idx_target == -1:
                break

        return (user, seq_mix, seq_source, seq_target, pos_target, neg_target,
                mix_idx_for_target, source_idx_for_target)

    while True:
        one_batch = []
        for _ in range(batch_size):
            one_batch.append(sample())
        result_queue.put(zip(*one_batch))


class WarpSampler(object):
    def __init__(self, source_name, target_name, cross_dataset, interval,
                 user_train_mix, user_train_source, user_train_target,
                 seq_idx_mix, seq_idx_source,
                 usernum, itemnum, seed, is_reversed=False,
                 batch_size=64, maxlen=10, n_workers=3):
        self.result_queue = Queue(maxsize=n_workers * 10)
        self.processors = []
        for _ in range(n_workers):
            self.processors.append(
                Process(target=sample_function, args=(
                    source_name, target_name, cross_dataset, interval,
                    user_train_mix, user_train_source, user_train_target,
                    seq_idx_mix, seq_idx_source,
                    usernum, itemnum, batch_size, maxlen,
                    self.result_queue, seed, is_reversed
                )))
            self.processors[-1].daemon = True
            self.processors[-1].start()

    def next_batch(self):
        return self.result_queue.get()

    def close(self):
        for p in self.processors:
            p.terminate()
            p.join()


def data_partition(source_name, target_name, cross_dataset, maxlen, data_root):
    """Load and partition Tri-CDR data for a cross-domain pair.

    Args:
        source_name: name of source domain (e.g., "Sports")
        target_name: name of target domain (e.g., "Clothing")
        cross_dataset: pair name (e.g., "asc")
        maxlen: maximum sequence length
        data_root: root data directory

    Returns:
        [user_train_mix, user_train_source, user_train_target,
         user_valid_target, user_test_target,
         seq_idx_mix, seq_idx_source,
         usernum, itemnum, interval]
    """
    subdir = f"{source_name}_{target_name}"
    data_dir = os.path.join(data_root, cross_dataset, subdir)

    # Load data files
    with open(os.path.join(data_dir, f"{source_name}_log_file_final.pkl"), 'rb') as f:
        source_log = pickle.load(f)
    with open(os.path.join(data_dir, f"{target_name}_log_file_final.pkl"), 'rb') as f:
        target_log = pickle.load(f)
    with open(os.path.join(data_dir, "mix_log_file_final.pkl"), 'rb') as f:
        mix_log = pickle.load(f)
    with open(os.path.join(data_dir, "item_index_mix.pkl"), 'rb') as f:
        item_index_mix = pickle.load(f)
    with open(os.path.join(data_dir, "user_index_overleap.pkl"), 'rb') as f:
        user_index_overlap = pickle.load(f)
    with open(os.path.join(data_dir, f"item_index_{source_name}.npy"), 'rb') as f:
        source_item_array = np.load(f)
    with open(os.path.join(data_dir, f"item_index_{target_name}.npy"), 'rb') as f:
        target_item_array = np.load(f)

    # Load metadata for interval/itemnum
    meta_path = os.path.join(data_dir, "meta.json")
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        interval = meta["interval"]
        itemnum = meta["itemnum"]
    else:
        interval = len(source_item_array)
        itemnum = interval + len(target_item_array)

    raw_user_ids = sorted(
        set(user_index_overlap.keys())
        & set(source_log.keys())
        & set(target_log.keys())
        & set(mix_log.keys())
    )
    user_id_map = {raw_uid: dense_uid for dense_uid, raw_uid in enumerate(raw_user_ids, start=1)}
    source_log = {user_id_map[uid]: source_log[uid] for uid in raw_user_ids}
    target_log = {user_id_map[uid]: target_log[uid] for uid in raw_user_ids}
    mix_log = {user_id_map[uid]: mix_log[uid] for uid in raw_user_ids}
    usernum = len(raw_user_ids)

    # Build training/validation/test splits
    user_train_mix = {}
    user_train_source = {}
    user_train_target = {}
    user_valid_target = {}
    user_test_target = {}
    seq_idx_mix = {}
    seq_idx_source = {}

    for k in range(1, usernum + 1):
        if k not in source_log or k not in target_log or k not in mix_log:
            continue

        v_mix = copy.deepcopy(mix_log[k])
        v_target = copy.deepcopy(target_log[k])

        # Find where target domain starts in the mix sequence
        target_last_id = v_target[-1]
        target_indices = [i for i, item in enumerate(v_mix) if item == target_last_id]
        if not target_indices:
            continue
        target_last_index = target_indices[-1]
        user_mix_truncated = v_mix[:target_last_index + 1]

        if len(user_mix_truncated) < 3:
            continue

        user_train_mix[k] = []
        user_train_source[k] = []
        user_train_target[k] = []
        user_valid_target[k] = []
        user_test_target[k] = []

        for item in reversed(user_mix_truncated):
            if item <= interval:  # source domain
                user_train_source[k].append(item)
                user_train_mix[k].append(item)
            else:  # target domain (interval+1 to itemnum)
                if len(user_test_target[k]) == 0:
                    user_test_target[k].append(item)
                elif len(user_valid_target[k]) == 0:
                    user_valid_target[k].append(item)
                elif len(user_test_target[k]) == 1 and len(user_valid_target[k]) == 1:
                    user_train_target[k].append(item)
                    user_train_mix[k].append(item)

        user_train_mix[k].reverse()
        user_train_source[k].reverse()
        user_train_target[k].reverse()

        # Build alignment indices
        pos_mix = len(user_train_mix[k]) - 1
        pos_source = len(user_train_source[k]) - 1
        mix_list = []
        source_list = []
        for item in reversed(user_train_mix[k]):
            if item <= interval:
                pos_source -= 1
            else:
                mix_list.append(pos_mix - 1)
                source_list.append(pos_source)
            pos_mix -= 1

        mix_for_target = mix_list[:-1]
        source_for_target = source_list[:-1]
        mix_for_target.reverse()
        source_for_target.reverse()

        seq_idx_mix[k] = [x - len(user_train_mix[k]) for x in mix_for_target]
        seq_idx_source[k] = [x - len(user_train_source[k]) for x in source_for_target]

    logging.info(f"data_partition: {usernum} users, interval={interval}, itemnum={itemnum}")
    return [user_train_mix, user_train_source, user_train_target,
            user_valid_target, user_test_target,
            seq_idx_mix, seq_idx_source,
            usernum, itemnum, interval]


def evaluate_SASRec(model, dataset, args):
    """Evaluate Tri-CDR model on test set."""
    with torch.no_grad():
        logging.info('Start evaluation...')
        [user_train_mix, user_train_source, user_train_target,
         user_valid_target, user_test_target,
         _, _, usernum, itemnum, interval] = dataset

        # Target domain items: interval+1..itemnum (our data always uses this range)
        random_min = interval + 1
        random_max = itemnum + 1

        item_entries = np.arange(start=random_min, stop=random_max, step=1, dtype=int)

        NDCG_1 = NDCG_5 = NDCG_10 = NDCG_20 = NDCG_50 = 0.0
        HT_1 = HT_5 = HT_10 = HT_20 = HT_50 = 0.0
        AUC = 0.0
        loss_total = 0.0
        valid_user = 0.0
        dist_ms = dist_mt = dist_st = 0.0

        for u in range(1, usernum + 1):
            if (u not in user_train_mix or u not in user_train_source or
                u not in user_train_target or u not in user_valid_target or
                u not in user_test_target):
                continue
            if (len(user_train_mix[u]) < 1 or len(user_train_source[u]) < 1 or
                len(user_train_target[u]) < 1 or len(user_valid_target[u]) < 1 or
                len(user_test_target[u]) < 1):
                continue

            seq_mix = np.zeros([args.maxlen], dtype=np.int32)
            seq_source = np.zeros([args.maxlen], dtype=np.int32)
            seq_target = np.zeros([args.maxlen], dtype=np.int32)

            idx_mix = args.maxlen - 1
            idx_source = args.maxlen - 1
            idx_target = args.maxlen - 1

            for item in reversed(user_train_mix[u]):
                seq_mix[idx_mix] = item
                idx_mix -= 1
                if idx_mix == -1:
                    break
            for item in reversed(user_train_source[u]):
                seq_source[idx_source] = item
                idx_source -= 1
                if idx_source == -1:
                    break

            seq_target[idx_target] = user_valid_target[u][0]
            idx_target -= 1
            for item in reversed(user_train_target[u]):
                seq_target[idx_target] = item
                idx_target -= 1
                if idx_target == -1:
                    break

            test_item = user_test_target[u][0]
            seen_target_items = set(user_train_target[u]) | set(user_valid_target[u])
            candidate_items = [item for item in item_entries
                               if item not in seen_target_items and item != test_item]
            item_idx = np.array([test_item] + candidate_items, dtype=int)
            labels = torch.zeros(len(item_idx), device=args.device)
            labels[0] = 1

            predictions, mix_feat, source_feat, target_feat = model.predict(
                *[np.array(l) for l in [[u], [seq_mix], [seq_source], [seq_target], [item_idx]]]
            )

            dist_ms += torch.dist(mix_feat, source_feat, p=2).item()
            dist_mt += torch.dist(mix_feat, target_feat, p=2).item()
            dist_st += torch.dist(source_feat, target_feat, p=2).item()

            AUC += roc_auc_score(labels.cpu().numpy(), predictions[0].cpu().detach().numpy())
            loss_test = torch.nn.BCEWithLogitsLoss()(predictions[0].detach(), labels)
            loss_total += loss_test.item()
            predictions = -predictions[0]
            rank = predictions.argsort().argsort()[0].item()
            valid_user += 1

            if rank < 1:
                NDCG_1 += 1 / np.log2(rank + 2); HT_1 += 1
            if rank < 5:
                NDCG_5 += 1 / np.log2(rank + 2); HT_5 += 1
            if rank < 10:
                NDCG_10 += 1 / np.log2(rank + 2); HT_10 += 1
            if rank < 20:
                NDCG_20 += 1 / np.log2(rank + 2); HT_20 += 1
            if rank < 50:
                NDCG_50 += 1 / np.log2(rank + 2); HT_50 += 1

            if valid_user % 1000 == 0:
                logging.info(f'  Processed {valid_user} test users')

    n = valid_user
    if n == 0:
        logging.warning("No valid users found during evaluation.")
        return (0.0, 0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0, 0.0,
                0.5, 0.0,
                0.0, 0.0, 0.0)
    return (NDCG_1 / n, NDCG_5 / n, NDCG_10 / n, NDCG_20 / n, NDCG_50 / n,
            HT_1 / n, HT_5 / n, HT_10 / n, HT_20 / n, HT_50 / n,
            AUC / n, loss_total / n,
            dist_ms / n, dist_mt / n, dist_st / n)
