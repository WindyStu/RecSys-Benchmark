"""
Build cross-domain datasets by merging two source domains with timestamp-aware
mixed sequences.

Output per dataset (data/{output}/):
  map_item.txt   — original_id \t mapped_item_id \t domain_label
  map_user.txt   — original_id \t mapped_user_id
  {output}.inter.json  — user interaction sequences
  {output}.item.json   — item metadata
"""

import argparse
import csv
import html
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path


# ---- Config ----

def build_cross_domain_configs(amazon_root, douban_root):
    amazon_root = Path(amazon_root)
    douban_root = Path(douban_root)
    return {
        "asc": {
            "a": {
                "name": "Sports",
                "reviews": str(amazon_root / "Sports_and_Outdoors_5.json"),
                "meta": str(amazon_root / "meta_Sports_and_Outdoors.json"),
                "format": "amazon",
            },
            "b": {
                "name": "Clothing",
                "reviews": str(amazon_root / "Clothing_Shoes_and_Jewelry_5.json"),
                "meta": str(amazon_root / "meta_Clothing_Shoes_and_Jewelry.json"),
                "format": "amazon",
            },
        },
        "ape": {
            "a": {
                "name": "Cell_Phones",
                "reviews": str(amazon_root / "Cell_Phones_and_Accessories_5.json"),
                "meta": str(amazon_root / "meta_Cell_Phones_and_Accessories.json"),
                "format": "amazon",
            },
            "b": {
                "name": "Electronics",
                "reviews": str(amazon_root / "Electronics_5.json"),
                "meta": str(amazon_root / "meta_Electronics.json"),
                "format": "amazon",
            },
        },
        "dbm": {
            "a": {
                "name": "Douban_Book",
                "reviews": str(douban_root / "bookreviews_cleaned.txt"),
                "meta": str(douban_root / "bookreviews_cleaned.txt"),
                "format": "douban",
            },
            "b": {
                "name": "Douban_Movie",
                "reviews": str(douban_root / "moviereviews_cleaned.txt"),
                "meta": str(douban_root / "moviereviews_cleaned.txt"),
                "format": "douban",
            },
        },
    }


# ---- Text cleaning ----

def clean_text(raw_text):
    """Consistent with data_preprocessing/utils.py:clean_text"""
    if raw_text is None:
        return ""
    if isinstance(raw_text, list):
        new_raw_text = []
        for raw in raw_text:
            raw = html.unescape(str(raw))
            raw = re.sub(r"</?\w+[^>]*>", "", raw)
            raw = re.sub(r'["\n\r]*', "", raw)
            new_raw_text.append(raw.strip())
        cleaned_text = " ".join(new_raw_text)
    else:
        if isinstance(raw_text, dict):
            cleaned_text = str(raw_text)[1:-1].strip()
        else:
            cleaned_text = str(raw_text).strip()
        cleaned_text = html.unescape(cleaned_text)
        cleaned_text = re.sub(r"</?\w+[^>]*>", "", cleaned_text)
        cleaned_text = re.sub(r'["\n\r]*', "", cleaned_text)

    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == ".":
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + "."
    else:
        cleaned_text = cleaned_text[:index] + "."

    if len(cleaned_text) >= 2000:
        cleaned_text = ""
    return cleaned_text


# ---- TSV helpers (for Douban) ----

def _safe_str(val):
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(str(x) for x in val if x)
    s = str(val).strip().strip('"')
    return s


def _parse_tsv(path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            cleaned = {_safe_str(k): _safe_str(v) for k, v in row.items()}
            rows.append(cleaned)
    return rows


# ---- Loaders ----

def load_amazon_domain(review_path, meta_path):
    """Load Amazon domain: returns (user_seq, item_meta)"""
    print(f"  Loading Amazon domain: {os.path.basename(review_path)}")

    # Metadata
    import ast
    item_meta = {}
    with open(meta_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = ast.literal_eval(line)
            except (ValueError, SyntaxError):
                continue
            asin = item.get("asin")
            if not asin:
                continue
            title = clean_text(item.get("title", ""))
            desc = clean_text(item.get("description", ""))
            item_meta[asin] = {"title": title, "description": desc}
    print(f"    Metadata: {len(item_meta)} items")

    # Reviews
    user_seq = defaultdict(list)  # user -> [(asin, timestamp)]
    with open(review_path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            uid = r.get("reviewerID")
            asin = r.get("asin")
            ts = r.get("unixReviewTime", 0)
            if uid and asin:
                user_seq[uid].append((asin, ts))
    print(f"    Reviews: {sum(len(v) for v in user_seq.values())} interactions, "
          f"{len(user_seq)} users")
    return dict(user_seq), item_meta


def load_douban_domain(review_path):
    """Load Douban domain: returns (user_seq, item_meta_from_labels)"""
    print(f"  Loading Douban domain: {os.path.basename(review_path)}")
    rows = _parse_tsv(review_path)

    user_seq = defaultdict(list)
    item_labels = defaultdict(set)

    for row in rows:
        uid = row.get("user_id", "").strip()
        iid = row.get("book_id", "").strip() or row.get("movie_id", "").strip()
        time_str = row.get("time", "").strip()
        labels = row.get("labels", "").strip()

        if not uid or not iid:
            continue

        try:
            ts = int(datetime.strptime(time_str, "%Y-%m-%d").timestamp())
        except (ValueError, OSError):
            ts = 0

        user_seq[uid].append((iid, ts))

        # Extract labels for metadata
        for tag in labels.split("|"):
            tag = tag.strip()
            if tag and not tag.isdigit():
                item_labels[iid].add(tag)

    item_meta = {}
    for iid, tags in item_labels.items():
        item_meta[iid] = {"title": "|".join(sorted(tags)), "description": ""}

    print(f"    Reviews: {sum(len(v) for v in user_seq.values())} interactions, "
          f"{len(user_seq)} users")
    print(f"    Labels metadata: {len(item_meta)} items")
    return dict(user_seq), item_meta


# ---- 5-core filter ----

def filter_5core(user2items):
    """Iterative 5-core filter."""
    iteration = 0
    while True:
        iteration += 1
        item_counts = defaultdict(int)
        for items in user2items.values():
            for item in items:
                item_counts[item] += 1
        valid_items = {item for item, count in item_counts.items() if count >= 5}
        new_u2i = {}
        for user, items in user2items.items():
            filtered = [item for item in items if item in valid_items]
            if len(filtered) >= 5:
                new_u2i[user] = filtered
        removed_u = len(user2items) - len(new_u2i)
        removed_i = len(item_counts) - len(valid_items)
        print(f"    Iter {iteration}: {len(new_u2i)} users, {len(valid_items)} items "
              f"(-{removed_u} / -{removed_i})")
        if removed_u == 0 and removed_i == 0:
            return new_u2i
        user2items = new_u2i


# ---- Main pipeline ----

def process_cross_domain(output_name, domain_a, domain_b, output_root):
    print(f"\n{'='*60}")
    print(f"Building: {output_name} ({domain_a['name']} + {domain_b['name']})")
    print(f"{'='*60}")

    # --- Load both domains ---
    user_seq = defaultdict(list)
    item_meta = {}

    for label, dom in [(0, domain_a), (1, domain_b)]:
        if dom["format"] == "amazon":
            u2i, meta = load_amazon_domain(dom["reviews"], dom["meta"])
        else:
            u2i, meta = load_douban_domain(dom["reviews"])

        # Tag items with domain label, and merge users
        for user, seq in u2i.items():
            for item_id, ts in seq:
                user_seq[user].append((item_id, ts, label))

        # Merge metadata (keep first occurrence wins for cross-domain duplicates)
        for item_id, info in meta.items():
            if item_id not in item_meta:
                item_meta[item_id] = info

    # --- Sort each user's sequence by timestamp ---
    print(f"\n  Merging sequences ...")
    for user in user_seq:
        user_seq[user].sort(key=lambda x: x[1])  # sort by timestamp
        user_seq[user] = [(item, label) for item, _, label in user_seq[user]]

    # --- 5-core filter ---
    # Build temporary dict with composite keys for filtering
    temp_u2i = {}
    for user, seq in user_seq.items():
        temp_u2i[user] = [f"{item}|{label}" for item, label in seq]

    print(f"  Before filter: {len(temp_u2i)} users, "
          f"{sum(len(v) for v in temp_u2i.values())} interactions")
    filtered = filter_5core(temp_u2i)

    # Rebuild after filter
    new_user_seq = {}
    for user, seq in filtered.items():
        new_seq = []
        for composite in seq:
            parts = composite.rsplit("|", 1)
            item_id = parts[0]
            label = int(parts[1])
            new_seq.append((item_id, label))
        new_user_seq[user] = new_seq

    user_seq = new_user_seq

    # --- Remap IDs ---
    print(f"\n  Remapping IDs ...")
    # Collect unique items with their domain labels
    item_label_map = {}
    for seq in user_seq.values():
        for item_id, label in seq:
            if item_id not in item_label_map:
                item_label_map[item_id] = label

    sorted_items = sorted(item_label_map.keys())
    sorted_users = sorted(user_seq.keys())

    item2new = {orig: i for i, orig in enumerate(sorted_items)}
    user2new = {orig: i for i, orig in enumerate(sorted_users)}

    # Write map_item.txt (JSON dict: {original_id: [mapped_id, domain_label]})
    output_dir = os.path.join(output_root, output_name)
    os.makedirs(output_dir, exist_ok=True)

    map_item = {}
    for orig_id in sorted_items:
        map_item[orig_id] = [item2new[orig_id], item_label_map[orig_id]]

    map_item_path = os.path.join(output_dir, "map_item.txt")
    with open(map_item_path, "w", encoding="utf-8") as f:
        json.dump(map_item, f, indent=4, ensure_ascii=False)
    print(f"  -> map_item.txt ({len(map_item)} items)")

    # Write map_user.txt (JSON dict: {original_user_id: mapped_user_id})
    map_user = {}
    for orig_id in sorted_users:
        map_user[orig_id] = user2new[orig_id]

    map_user_path = os.path.join(output_dir, "map_user.txt")
    with open(map_user_path, "w", encoding="utf-8") as f:
        json.dump(map_user, f, indent=4, ensure_ascii=False)
    print(f"  -> map_user.txt ({len(map_user)} users)")

    # Write inter.json
    inter = {}
    for user in sorted_users:
        uid = user2new[user]
        inter[str(uid)] = [item2new[item_id] for item_id, _ in user_seq[user]]

    inter_path = os.path.join(output_dir, f"{output_name}.inter.json")
    with open(inter_path, "w", encoding="utf-8") as f:
        json.dump(inter, f, indent=4, ensure_ascii=False)
    print(f"  -> {output_name}.inter.json ({len(inter)} users)")

    # Write item.json
    items = {}
    for orig_id in sorted_items:
        iid = item2new[orig_id]
        if orig_id in item_meta:
            items[str(iid)] = item_meta[orig_id]
        else:
            items[str(iid)] = {"title": "", "description": ""}

    item_path = os.path.join(output_dir, f"{output_name}.item.json")
    with open(item_path, "w", encoding="utf-8") as f:
        json.dump(items, f, indent=4, ensure_ascii=False)
    print(f"  -> {output_name}.item.json ({len(items)} items)")

    # Stats
    seq_lens = [len(v) for v in inter.values()]
    total_inter = sum(seq_lens)
    avg_len = total_inter / len(seq_lens) if seq_lens else 0
    domain_a_count = sum(1 for v in item_label_map.values() if v == 0)
    domain_b_count = sum(1 for v in item_label_map.values() if v == 1)
    print(f"\n  [{output_name}] Users: {len(inter)}, Items: {len(items)}, "
          f"Interactions: {total_inter}, Avg len: {avg_len:.1f}")
    print(f"  Domain 0 ({domain_a['name']}): {domain_a_count} items")
    print(f"  Domain 1 ({domain_b['name']}): {domain_b_count} items")


# ---- CLI ----

def parse_args():
    parser = argparse.ArgumentParser(
        description="Build mixed-sequence cross-domain datasets for the benchmark."
    )
    parser.add_argument("--name", type=str, default="",
                        choices=["asc", "ape", "dbm", ""],
                        help="Cross-domain dataset name. Omit to build all three.")
    parser.add_argument("--output_root", type=str, default="data",
                        help="Output root directory.")
    parser.add_argument("--amazon_root", type=str, default=os.getenv("AMAZON_RAW_ROOT", ""),
                        help="Amazon raw data directory; defaults to AMAZON_RAW_ROOT.")
    parser.add_argument("--douban_root", type=str, default=os.getenv("DOUBAN_RAW_ROOT", ""),
                        help="Douban raw data directory; defaults to DOUBAN_RAW_ROOT.")
    return parser.parse_args()


def main():
    args = parse_args()

    output_root = os.path.abspath(args.output_root)
    if args.name in ("asc", "ape") and not args.amazon_root:
        raise ValueError("Set AMAZON_RAW_ROOT or pass --amazon_root.")
    if args.name == "dbm" and not args.douban_root:
        raise ValueError("Set DOUBAN_RAW_ROOT or pass --douban_root.")
    if not args.name and (not args.amazon_root or not args.douban_root):
        raise ValueError("Building all datasets requires AMAZON_RAW_ROOT and DOUBAN_RAW_ROOT.")
    configs = build_cross_domain_configs(args.amazon_root, args.douban_root)

    if args.name:
        cfg = configs[args.name]
        process_cross_domain(args.name, cfg["a"], cfg["b"], output_root)
    else:
        for name, cfg in configs.items():
            process_cross_domain(name, cfg["a"], cfg["b"], output_root)

    print(f"\n{'='*60}")
    print("All cross-domain datasets built.")
    print(f"Next: run text embedding for each:")
    print(f"  python data_preprocessing/aliyun_text_emb.py --dataset asc --root data --field_mode join")
    print(f"  python data_preprocessing/aliyun_text_emb.py --dataset ape --root data --field_mode join")
    print(f"  python data_preprocessing/aliyun_text_emb.py --dataset dbm --root data --field_mode join")


if __name__ == "__main__":
    main()
