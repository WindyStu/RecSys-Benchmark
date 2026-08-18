"""
Preprocess raw review + item metadata files into LETTER data format.

Supports:
  - Amazon 5-core  (JSON Lines / Python literal)
  - Douban          (TSV)

Output:
    data/{Dataset}/
        {Dataset}.item.json    # {"item_id": {"title": "...", "description": "..."}}
        {Dataset}.inter.json   # {"user_id": [item_id, item_id, ...]}
"""

import argparse
import ast
import collections
import csv
import html
import json
import os
import re
import sys
from collections import defaultdict
from datetime import datetime


# ---- Dataset name mappings ----

# Amazon: review / meta filename base -> short output directory name
DATASET_MAP_AMAZON = {
    "Cell_Phones_and_Accessories": "Cell_Phones",
    "Clothing_Shoes_and_Jewelry": "Clothing",
    "Electronics": "Electronics",
    "Sports_and_Outdoors": "Sports",
}

# Douban: file prefix -> short output directory name
DATASET_MAP_DOUBAN = {
    "book": "Douban_Book",
    "movie": "Douban_Movie",
}

# ---- Text cleaning (consistent with data_preprocessing/utils.py:clean_text) ----


def clean_text(raw_text):
    """Clean HTML entities, tags, quotes, newlines. Return empty string if too long."""
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

    # Ensure ends with period
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


# ================================================================
#  Douban loaders (TSV format)
# ================================================================

def _safe_str(val):
    """Convert a value to a stripped string, handling None, lists, etc."""
    if val is None:
        return ""
    if isinstance(val, list):
        return " ".join(str(x) for x in val if x)
    s = str(val).strip().strip('"')
    return s


def _parse_tsv(path):
    """Read a TSV file with header row, return list of dicts."""
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=csv.QUOTE_NONE)
        for row in reader:
            cleaned = {}
            for k, v in row.items():
                key = _safe_str(k)
                cleaned[key] = _safe_str(v)
            rows.append(cleaned)
    return rows


def load_metadata_douban(items_path, item_id_col, title_col=None, desc_col=None):
    """
    Parse Douban items TSV file.
    Returns: dict {item_id_str: {"title": str, "description": str}}
    """
    print(f"  Loading metadata: {os.path.basename(items_path)} ...")
    meta = {}
    rows = _parse_tsv(items_path)
    for row in rows:
        iid = row.get(item_id_col, "").strip()
        if not iid:
            continue
        title = clean_text(row.get(title_col, "")) if title_col else iid
        desc = clean_text(row.get(desc_col, "")) if desc_col else ""
        meta[iid] = {"title": title, "description": desc}

    print(f"    Loaded {len(meta)} items")
    return meta


def load_reviews_douban(review_path, user_col, item_col, time_col):
    """
    Parse Douban reviews TSV file.
    Returns: dict {user_id: [(item_id, sort_key), ...]}
    """
    print(f"  Loading reviews: {os.path.basename(review_path)} ...")
    user2items = defaultdict(list)
    review_count = 0
    rows = _parse_tsv(review_path)
    for row in rows:
        user = row.get(user_col, "").strip()
        item = row.get(item_col, "").strip()
        time_str = row.get(time_col, "").strip()
        if not user or not item:
            continue
        # Parse time string (YYYY-MM-DD) to integer key for sorting,
        # fall back to index position on parse failure
        try:
            time_key = int(datetime.strptime(time_str, "%Y-%m-%d").timestamp())
        except (ValueError, OSError):
            time_key = review_count
        user2items[user].append((item, time_key))
        review_count += 1

    # Sort and strip timestamps
    for user in user2items:
        user2items[user].sort(key=lambda x: x[1])
        user2items[user] = [item for item, _ in user2items[user]]

    print(f"    Loaded {review_count} reviews, {len(user2items)} users")
    return dict(user2items)


def extract_metadata_from_labels(review_path, item_col, labels_col):
    """
    Extract item metadata from Douban review labels.
    Labels format: |tag1|12345|tag2|67890|  (numbers are filtered out)
    Returns: dict {item_id_str: {"title": str, "description": str}}
    """
    print(f"  Extracting labels from: {os.path.basename(review_path)} ...")
    item2labels = defaultdict(set)
    rows = _parse_tsv(review_path)
    for row in rows:
        item = row.get(item_col, "").strip()
        labels_raw = row.get(labels_col, "").strip()
        if not item or not labels_raw:
            continue
        # Split by | and filter: keep non-empty, non-numeric entries
        for tag in labels_raw.split("|"):
            tag = tag.strip()
            if not tag:
                continue
            # Drop pure-numeric tags (e.g. "8956")
            if tag.isdigit():
                continue
            item2labels[item].add(tag)

    # Build meta: join all unique labels per item (no clean_text — labels are clean tags)
    meta = {}
    for item, tags in item2labels.items():
        label_str = "|".join(sorted(tags))
        meta[item] = {"title": label_str, "description": ""}

    print(f"    Extracted metadata for {len(meta)} items")
    return meta


# ================================================================
#  Amazon loaders (JSON Lines / Python literal)
# ================================================================

# ---- Step 1: Load metadata ----

def load_metadata(meta_path):
    """
    Parse meta_*.json (Python literal format) line by line.
    Returns: dict {asin: {"title": str, "description": str}}
    """
    print(f"  Loading metadata: {os.path.basename(meta_path)} ...")
    meta = {}
    skipped = 0
    with open(meta_path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                item = ast.literal_eval(line)
            except (ValueError, SyntaxError) as e:
                skipped += 1
                if skipped <= 5:
                    print(f"    [WARN] Parse error at line {line_no}: {e}")
                continue

            asin = item.get("asin")
            if not asin:
                skipped += 1
                continue

            title = clean_text(item.get("title", ""))
            description = clean_text(item.get("description", ""))

            meta[asin] = {"title": title, "description": description}

    print(f"    Loaded {len(meta)} items ({skipped} skipped)")
    return meta


# ---- Step 2: Load reviews ----

def load_reviews(review_path):
    """
    Parse *_5.json (JSON Lines) line by line.
    Returns: dict {reviewerID: [(asin, unixReviewTime), ...]}
    """
    print(f"  Loading reviews: {os.path.basename(review_path)} ...")
    user2items = defaultdict(list)
    review_count = 0
    skipped = 0
    with open(review_path, "r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                review = json.loads(line)
            except json.JSONDecodeError as e:
                skipped += 1
                if skipped <= 5:
                    print(f"    [WARN] JSON parse error at line {line_no}: {e}")
                continue

            reviewer = review.get("reviewerID")
            asin = review.get("asin")
            timestamp = review.get("unixReviewTime", 0)

            if not reviewer or not asin:
                skipped += 1
                continue

            user2items[reviewer].append((asin, timestamp))
            review_count += 1

    # Sort each user's sequence by timestamp
    for reviewer in user2items:
        user2items[reviewer].sort(key=lambda x: x[1])
        # Keep only asin, drop timestamp
        user2items[reviewer] = [asin for asin, _ in user2items[reviewer]]

    print(f"    Loaded {review_count} reviews, {len(user2items)} users ({skipped} skipped)")
    return dict(user2items)


# ---- Step 3: 5-core filter ----

def filter_5core(user2items):
    """
    Iteratively remove users with < 5 interactions and items appearing < 5 times,
    until convergence.
    Returns: filtered user2items dict.
    """
    print(f"  Applying 5-core filter ...")
    iteration = 0
    while True:
        iteration += 1
        # Count item frequencies
        item_counts = defaultdict(int)
        for items in user2items.values():
            for item in items:
                item_counts[item] += 1

        # Filter: keep items appearing >= 5 times
        valid_items = {item for item, count in item_counts.items() if count >= 5}

        # Filter: keep users with >= 5 valid items
        new_user2items = {}
        for user, items in user2items.items():
            filtered = [item for item in items if item in valid_items]
            if len(filtered) >= 5:
                new_user2items[user] = filtered

        removed_users = len(user2items) - len(new_user2items)
        removed_items = len(item_counts) - len(valid_items)

        print(f"    Iteration {iteration}: {len(new_user2items)} users, "
              f"{len(valid_items)} items "
              f"(-{removed_users} users, -{removed_items} items)")

        if removed_users == 0 and removed_items == 0:
            break

        user2items = new_user2items

    return new_user2items


# ---- Step 4: Remap to dense IDs ----

def remap_ids(user2items, meta):
    """
    Assign dense 0-based IDs to users and items.
    Returns: (inter_dict, item_dict)
        inter_dict: {"0": [0, 1, ...], ...}
        item_dict: {"0": {"title": "...", "description": "..."}, ...}
    """
    # Collect all remaining items
    all_items = set()
    for items in user2items.values():
        all_items.update(items)

    # Sort for deterministic ID assignment
    sorted_items = sorted(all_items)
    sorted_users = sorted(user2items.keys())

    item2id = {asin: str(i) for i, asin in enumerate(sorted_items)}
    user2id = {user: str(i) for i, user in enumerate(sorted_users)}

    # Build inter.json
    inter_dict = {}
    for user in sorted_users:
        uid = user2id[user]
        inter_dict[uid] = [int(item2id[asin]) for asin in user2items[user]]

    # Build item.json (only for items in interactions)
    item_dict = {}
    for asin in sorted_items:
        iid = item2id[asin]
        if asin in meta:
            item_dict[iid] = meta[asin]
        else:
            # ASIN has no metadata entry
            item_dict[iid] = {"title": "", "description": ""}

    print(f"    Remapped: {len(inter_dict)} users, {len(item_dict)} items")
    return inter_dict, item_dict


# ---- Step 5: Write output ----

def write_output(dataset_name, inter_dict, item_dict, output_root):
    """Write inter.json and item.json to the output directory."""
    output_dir = os.path.join(output_root, dataset_name)
    os.makedirs(output_dir, exist_ok=True)

    inter_path = os.path.join(output_dir, f"{dataset_name}.inter.json")
    item_path = os.path.join(output_dir, f"{dataset_name}.item.json")

    print(f"  Writing output to: {output_dir}/")
    with open(inter_path, "w", encoding="utf-8") as f:
        json.dump(inter_dict, f, indent=4, ensure_ascii=False)
    print(f"    -> {dataset_name}.inter.json ({len(inter_dict)} users)")

    with open(item_path, "w", encoding="utf-8") as f:
        json.dump(item_dict, f, indent=4, ensure_ascii=False)
    print(f"    -> {dataset_name}.item.json ({len(item_dict)} items)")


# ---- Statistics ----

def print_stats(name, user2items, item_dict):
    """Print summary statistics for a dataset."""
    seq_lens = [len(items) for items in user2items.values()]
    avg_len = sum(seq_lens) / len(seq_lens) if seq_lens else 0
    print(f"  [{name}] Users: {len(user2items)}, Items: {len(item_dict)}, "
          f"Interactions: {sum(seq_lens)}, Avg seq len: {avg_len:.1f}")


# ---- Main pipeline ----

def process_dataset(base_name, output_name, data_root, output_root):
    """Run the full pipeline for one dataset."""
    print(f"\n{'='*60}")
    print(f"Processing: {base_name} -> {output_name}")
    print(f"{'='*60}")

    review_path = os.path.join(data_root, f"{base_name}_5.json")
    meta_path = os.path.join(data_root, f"meta_{base_name}.json")

    if not os.path.exists(review_path):
        print(f"  [SKIP] Review file not found: {review_path}")
        return
    if not os.path.exists(meta_path):
        print(f"  [SKIP] Metadata file not found: {meta_path}")
        return

    # Step 1: Load metadata
    meta = load_metadata(meta_path)

    # Step 2: Load reviews
    user2items = load_reviews(review_path)

    # Step 3: 5-core filter
    stats_before = len(user2items)
    user2items = filter_5core(user2items)
    stats_after = len(user2items)

    # Step 4: Remap to dense IDs
    inter_dict, item_dict = remap_ids(user2items, meta)

    # Step 5: Write output
    write_output(output_name, inter_dict, item_dict, output_root)

    # Stats
    print_stats(output_name, inter_dict, item_dict)
    print(f"  Users filtered: {stats_before} -> {stats_after} "
          f"(-{stats_before - stats_after})")


# ---- Validation ----

def validate_output(dataset_name, output_root):
    """Validate the output files for a dataset."""
    output_dir = os.path.join(output_root, dataset_name)
    inter_path = os.path.join(output_dir, f"{dataset_name}.inter.json")
    item_path = os.path.join(output_dir, f"{dataset_name}.item.json")

    if not os.path.exists(inter_path) or not os.path.exists(item_path):
        print(f"  [VALIDATE] Missing output files for {dataset_name}")
        return False

    with open(inter_path, "r", encoding="utf-8") as f:
        inter = json.load(f)
    with open(item_path, "r", encoding="utf-8") as f:
        items = json.load(f)

    errors = []

    # Check item IDs are dense and 0-based
    item_ids = sorted(int(k) for k in items.keys())
    if item_ids != list(range(len(item_ids))):
        errors.append(f"Item IDs are not dense 0-based: expected 0..{len(item_ids)-1}")

    # Check 5-core constraint
    item_appearances = defaultdict(int)
    for uid, seq in inter.items():
        if len(seq) < 5:
            errors.append(f"User {uid} has only {len(seq)} interactions (< 5)")
        for iid in seq:
            item_appearances[str(iid)] += 1

    for iid in item_ids:
        if str(iid) not in item_appearances:
            errors.append(f"Item {iid} exists in item.json but NOT in inter.json")
    for iid_str, count in item_appearances.items():
        if count < 5:
            errors.append(f"Item {iid_str} appears only {count} times (< 5)")

    # Check all items in inter.json exist in item.json
    for uid, seq in inter.items():
        for iid in seq:
            if str(iid) not in items:
                errors.append(f"Item {iid} from inter.json missing in item.json")

    # Check text field presence
    missing_title = sum(1 for v in items.values() if not v.get("title"))
    missing_desc = sum(1 for v in items.values() if not v.get("description"))
    if missing_title > 0:
        print(f"  [INFO] {missing_title}/{len(items)} items have empty title")
    if missing_desc > 0:
        print(f"  [INFO] {missing_desc}/{len(items)} items have empty description")

    if errors:
        print(f"  [VALIDATE] {len(errors)} errors found for {dataset_name}:")
        for e in errors[:10]:
            print(f"    - {e}")
        if len(errors) > 10:
            print(f"    ... and {len(errors) - 10} more")
        return False
    else:
        print(f"  [VALIDATE] {dataset_name}: OK")
        return True


def process_dataset_douban(prefix, output_name, data_root, output_root,
                           item_id_col, title_col, desc_col,
                           user_col, item_col, time_col, labels_col="labels"):
    """Run the full pipeline for one Douban dataset.

    Metadata is extracted from the labels field in reviews (| separated,
    numeric entries filtered out). Falls back to items TSV if labels unavailable.
    """
    print(f"\n{'='*60}")
    print(f"Processing: {prefix} -> {output_name}  (Douban TSV)")
    print(f"{'='*60}")

    items_path = os.path.join(data_root, f"{prefix}s_cleaned.txt")
    review_path = os.path.join(data_root, f"{prefix}reviews_cleaned.txt")

    if not os.path.exists(review_path):
        print(f"  [SKIP] Review file not found: {review_path}")
        return

    # Step 1: Extract metadata from review labels (primary method)
    meta = extract_metadata_from_labels(review_path, item_col, labels_col)

    # Step 2: Load reviews for user sequences
    user2items = load_reviews_douban(review_path, user_col, item_col, time_col)

    # Step 3: 5-core filter
    stats_before = len(user2items)
    user2items = filter_5core(user2items)
    stats_after = len(user2items)

    # Step 4: Remap
    inter_dict, item_dict = remap_ids(user2items, meta)

    # Step 5: Write
    write_output(output_name, inter_dict, item_dict, output_root)

    # Stats
    print_stats(output_name, inter_dict, item_dict)
    print(f"  Users filtered: {stats_before} -> {stats_after} "
          f"(-{stats_before - stats_after})")


# ---- CLI ----

def parse_args():
    parser = argparse.ArgumentParser(
        description="Preprocess Amazon 5-core / Douban data into benchmark JSON format."
    )
    parser.add_argument(
        "--data_root", type=str, default=os.getenv("RECSYS_RAW_DATA_ROOT", ""),
        help="Directory containing raw review and metadata files; defaults to RECSYS_RAW_DATA_ROOT."
    )
    parser.add_argument(
        "--output_root", type=str, default="data",
        help="Output root directory (default: data/)."
    )
    parser.add_argument(
        "--format", type=str, choices=["amazon", "douban"], default="amazon",
        help="Input data format (default: amazon)."
    )
    parser.add_argument(
        "--dataset", type=str, default="",
        help="Process a single dataset. Amazon: base name (e.g. 'Cell_Phones_and_Accessories'). "
             "Douban: prefix (e.g. 'book', 'movie'). Omit to process all."
    )
    parser.add_argument(
        "--validate_only", action="store_true",
        help="Only validate existing output files, skip processing."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if not args.data_root:
        raise ValueError("Set RECSYS_RAW_DATA_ROOT or pass --data_root.")

    # Resolve absolute paths
    data_root = os.path.abspath(args.data_root)
    output_root = os.path.abspath(args.output_root)
    print(f"Data root:   {data_root}")
    print(f"Output root: {output_root}")
    print(f"Format:      {args.format}")

    if args.format == "douban":
        _main_douban(args, data_root, output_root)
    else:
        _main_amazon(args, data_root, output_root)


def _main_amazon(args, data_root, output_root):
    dataset_map = DATASET_MAP_AMAZON

    if args.validate_only:
        for output_name in dataset_map.values():
            print(f"\nValidating: {output_name}")
            validate_output(output_name, output_root)
        return

    if args.dataset:
        base_name = args.dataset
        if base_name not in dataset_map:
            print(f"Unknown dataset: {base_name}")
            print(f"Available: {list(dataset_map.keys())}")
            sys.exit(1)
        output_name = dataset_map[base_name]
        process_dataset(base_name, output_name, data_root, output_root)
        print(f"\n{'='*60}")
        print("Validation")
        print(f"{'='*60}")
        validate_output(output_name, output_root)
    else:
        for base_name, output_name in dataset_map.items():
            process_dataset(base_name, output_name, data_root, output_root)
        print(f"\n{'='*60}")
        print("Validation")
        print(f"{'='*60}")
        all_ok = True
        for output_name in dataset_map.values():
            if not validate_output(output_name, output_root):
                all_ok = False
        if all_ok:
            print("\nAll datasets processed and validated successfully!")
        else:
            print("\nSome datasets have validation errors. See above.")


def _main_douban(args, data_root, output_root):
    """Douban datasets with their column mappings."""
    douban_configs = {
        "book": {
            "output_name": "Douban_Book",
            "item_id_col": "book_id",
            "title_col": None,        # no title → use book_id itself
            "desc_col": None,         # no description
            "user_col": "user_id",
            "item_col": "book_id",
            "time_col": "time",
        },
        "movie": {
            "output_name": "Douban_Movie",
            "item_id_col": "movie_id",
            "title_col": "name",
            "desc_col": "summary",
            "user_col": "user_id",
            "item_col": "movie_id",
            "time_col": "time",
        },
    }

    if args.validate_only:
        for cfg in douban_configs.values():
            print(f"\nValidating: {cfg['output_name']}")
            validate_output(cfg["output_name"], output_root)
        return

    if args.dataset:
        if args.dataset not in douban_configs:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {list(douban_configs.keys())}")
            sys.exit(1)
        cfg = douban_configs[args.dataset]
        process_dataset_douban(
            prefix=args.dataset,
            output_name=cfg["output_name"],
            data_root=data_root,
            output_root=output_root,
            item_id_col=cfg["item_id_col"],
            title_col=cfg["title_col"],
            desc_col=cfg["desc_col"],
            user_col=cfg["user_col"],
            item_col=cfg["item_col"],
            time_col=cfg["time_col"],
        )
        print(f"\n{'='*60}")
        print("Validation")
        print(f"{'='*60}")
        validate_output(cfg["output_name"], output_root)
    else:
        for prefix, cfg in douban_configs.items():
            process_dataset_douban(
                prefix=prefix,
                output_name=cfg["output_name"],
                data_root=data_root,
                output_root=output_root,
                item_id_col=cfg["item_id_col"],
                title_col=cfg["title_col"],
                desc_col=cfg["desc_col"],
                user_col=cfg["user_col"],
                item_col=cfg["item_col"],
                time_col=cfg["time_col"],
            )
        print(f"\n{'='*60}")
        print("Validation")
        print(f"{'='*60}")
        all_ok = True
        for cfg in douban_configs.values():
            if not validate_output(cfg["output_name"], output_root):
                all_ok = False
        if all_ok:
            print("\nAll Douban datasets processed and validated successfully!")
        else:
            print("\nSome datasets have validation errors. See above.")


if __name__ == "__main__":
    main()
