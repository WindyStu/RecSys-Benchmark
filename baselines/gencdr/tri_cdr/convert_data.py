"""
convert_data.py — Convert GenCDR .inter.json + map_item.txt to Tri-CDR .pkl format.

For each cross-domain pair, produces Tri-CDR compatible data files in a subdirectory
named {source}_{target}/ (or {target}_{source}/ for the reverse direction).

Usage:
    python convert_data.py --pair asc
    python convert_data.py --pair asc --direction 0to1   # Sports → Clothing only
    python convert_data.py --pair dbm --direction both   # both directions
    python convert_data.py --all
"""

import os
import sys
import json
import pickle
import argparse
import logging
import numpy as np

# ---- Config ----
DATA_ROOT = "/nfsshare/home/liujingyan/data/CDSR/data"

# Domain label mapping: label 0 → source, label 1 → target
PAIR_DOMAINS = {
    "asc":  {"0": "Sports", "1": "Clothing"},
    "ape":  {"0": "Phone", "1": "Electronics"},
    "dbm":  {"0": "Book", "1": "Movies"},
    "ghk":  {"0": "Grocery", "1": "Home_Kitchen"},
}


def load_raw_data(pair_name, data_root):
    """Load inter.json and map_item.txt for a given pair."""
    pair_dir = os.path.join(data_root, pair_name)
    inter_path = os.path.join(pair_dir, f"{pair_name}.inter.json")
    map_item_path = os.path.join(pair_dir, "map_item.txt")

    with open(inter_path, 'r') as f:
        inter = json.load(f)
    with open(map_item_path, 'r') as f:
        map_item = json.load(f)

    logging.info(f"Loaded {pair_name}: {len(inter)} users, {len(map_item)} items")
    return inter, map_item


def build_item_mapping(map_item, source_label=0, target_label=1):
    """Build source and target item ID sets and global remapping.

    map_item format: {original_item_id: [item_index, domain_label]}
    item_index is the integer used in inter.json.
    """
    source_items = set()
    target_items = set()

    for orig_id, labels in map_item.items():
        item_idx = int(labels[0])  # index used in inter.json
        domain_label = int(labels[1])
        if domain_label == source_label:
            source_items.add(item_idx)
        elif domain_label == target_label:
            target_items.add(item_idx)

    source_sorted = sorted(source_items)
    target_sorted = sorted(target_items)

    interval = len(source_sorted)
    itemnum = interval + len(target_sorted)

    old2new = {}
    new2old_source = [None]  # index 0 unused (SASRec convention: 0 = padding)
    new2old_target = [None]

    for new_id, old_idx in enumerate(source_sorted, start=1):
        old2new[old_idx] = new_id
        new2old_source.append(str(old_idx))

    for new_id, old_idx in enumerate(target_sorted, start=interval + 1):
        old2new[old_idx] = new_id
        new2old_target.append(str(old_idx))

    logging.info(f"  Source items: {len(source_sorted)}, Target items: {len(target_sorted)}")
    logging.info(f"  Interval: {interval}, Itemnum: {itemnum}")
    return source_items, target_items, old2new, new2old_source, new2old_target, interval, itemnum


def build_user_sequences(inter, source_items, target_items, old2new, interval):
    """Build per-user mix/source/target sequences with global item IDs."""
    user_mix = {}
    user_source = {}
    user_target = {}

    for user_idx, item_list in inter.items():
        mix_seq = [old2new[item] for item in item_list if item in old2new]
        source_seq = [old2new[item] for item in item_list if item in source_items]
        target_seq = [old2new[item] for item in item_list if item in target_items]

        # Only keep users active in BOTH domains
        if len(source_seq) >= 1 and len(target_seq) >= 1:
            uid = int(user_idx) + 1  # SASRec convention: user IDs start from 1
            user_mix[uid] = mix_seq
            user_source[uid] = source_seq
            user_target[uid] = target_seq

    overlap_count = len(user_mix)
    logging.info(f"  Overlapping users: {overlap_count}")
    return user_mix, user_source, user_target


def save_pkl_files(output_dir, source_name, target_name,
                   user_mix, user_source, user_target,
                   new2old_source, new2old_target, interval):
    """Save all Tri-CDR format pkl files."""
    os.makedirs(output_dir, exist_ok=True)

    # User sequence files
    _save_pkl(os.path.join(output_dir, f"{source_name}_log_file_final.pkl"), user_source)
    _save_pkl(os.path.join(output_dir, f"{target_name}_log_file_final.pkl"), user_target)
    _save_pkl(os.path.join(output_dir, "mix_log_file_final.pkl"), user_mix)

    # Item index dictionaries
    itemnum = interval + len(new2old_target) - 1
    item_index_source = {name: idx for idx, name in enumerate(new2old_source) if name is not None and idx > 0}
    item_index_target = {name: idx for idx, name in enumerate(new2old_target) if name is not None and idx > 0}

    # Rebuild with 1-based indexing matching global IDs
    item_index_source = {}
    for idx, name in enumerate(new2old_source):
        if name is not None and idx > 0:
            item_index_source[name] = idx

    item_index_target = {}
    for idx, name in enumerate(new2old_target):
        if name is not None and idx > 0:
            item_index_target[name] = idx

    # Mix index: all items (source + target)
    item_index_mix = {}
    item_index_mix.update(item_index_source)
    item_index_mix.update(item_index_target)

    _save_pkl(os.path.join(output_dir, f"item_index_{source_name}.pkl"), item_index_source)
    _save_pkl(os.path.join(output_dir, f"item_index_{target_name}.pkl"), item_index_target)
    _save_pkl(os.path.join(output_dir, "item_index_mix.pkl"), item_index_mix)

    # Numpy arrays (used in data_partition for name lookup)
    np.save(os.path.join(output_dir, f"item_index_{source_name}.npy"),
            np.array(new2old_source[1:]))  # skip None at index 0
    np.save(os.path.join(output_dir, f"item_index_{target_name}.npy"),
            np.array(new2old_target[1:]))

    # Overlapping users
    user_index_overlap = {uid: True for uid in user_mix.keys()}
    _save_pkl(os.path.join(output_dir, "user_index_overleap.pkl"), user_index_overlap)

    # Save metadata
    meta = {
        "pair": os.path.basename(os.path.dirname(output_dir)),
        "source_name": source_name, "target_name": target_name,
        "interval": interval, "itemnum": itemnum,
        "num_users": len(user_mix), "num_items": itemnum
    }
    with open(os.path.join(output_dir, "meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    logging.info(f"  Saved {9} files to {output_dir}")


def _save_pkl(path, data):
    with open(path, 'wb') as f:
        pickle.dump(data, f)


def process_pair(pair_name, data_root, direction="both"):
    """Convert one pair's data to Tri-CDR format."""
    inter, map_item = load_raw_data(pair_name, data_root)
    domains = PAIR_DOMAINS[pair_name]

    directions = []
    if direction in ("both", "0to1"):
        directions.append((0, 1, domains["0"], domains["1"]))
    if direction in ("both", "1to0"):
        directions.append((1, 0, domains["1"], domains["0"]))

    for src_label, tgt_label, src_name, tgt_name in directions:
        logging.info(f"{'='*50}")
        logging.info(f"  Direction: {src_name} → {tgt_name} (label {src_label}→{tgt_label})")

        source_items, target_items, old2new, n2o_src, n2o_tgt, interval, itemnum = \
            build_item_mapping(map_item, source_label=src_label, target_label=tgt_label)

        user_mix, user_source, user_target = build_user_sequences(
            inter, source_items, target_items, old2new, interval)

        out_dir = os.path.join(data_root, pair_name, f"{src_name}_{tgt_name}")
        save_pkl_files(out_dir, src_name, tgt_name,
                       user_mix, user_source, user_target,
                       n2o_src, n2o_tgt, interval)

    return True


def main():
    parser = argparse.ArgumentParser(description="Convert GenCDR data to Tri-CDR format")
    parser.add_argument("--pair", type=str, default=None, help="Single pair to convert")
    parser.add_argument("--all", action="store_true", help="Convert all pairs")
    parser.add_argument("--direction", type=str, default="both",
                        choices=["both", "0to1", "1to0"],
                        help="Conversion direction (default: both)")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT,
                        help=f"Data root directory (default: {DATA_ROOT})")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s - %(levelname)s - %(message)s')

    if args.all:
        pairs = list(PAIR_DOMAINS.keys())
    elif args.pair:
        pairs = [args.pair]
    else:
        parser.print_help()
        return

    for pair in pairs:
        logging.info(f"Converting pair: {pair}")
        try:
            process_pair(pair, args.data_root, args.direction)
        except Exception as e:
            logging.error(f"Failed to convert {pair}: {e}", exc_info=True)

    logging.info("All done!")


if __name__ == "__main__":
    main()
