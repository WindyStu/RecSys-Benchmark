import argparse
import json
from pathlib import Path


DEFAULT_SINGLE_NAMES = {
    "Beauty": "abeauty",
    "Electronics": "ae",
    "Grocery": "af",
    "Home_Kitchen": "ak",
    "Douban_Movie": "am",
    "Douban_Book": "abook",
    "Cell_Phones": "cell_phones",
    "Clothing": "clothing",
    "Instruments": "instruments",
    "Sports": "sports",
    "Yelp": "yelp",
}

DEFAULT_PAIR_NAMES = {
    ("Beauty", "Electronics"): "abe",
    ("Grocery", "Home_Kitchen"): "afk",
    ("Douban_Movie", "Douban_Book"): "amb",
}


def _sort_key(value):
    text = str(value)
    return (0, int(text)) if text.isdigit() else (1, text)


def _load_inter(source_root, domain):
    path = Path(source_root) / domain / f"{domain}.inter.json"
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)
    return {str(user): [int(item) for item in items] for user, items in raw.items()}


def _write_json(path, data):
    with Path(path).open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=True)


def _write_sequences(path, sequences):
    with Path(path).open("w", encoding="utf-8") as f:
        for user_idx, sequence in sequences:
            parts = [str(user_idx)]
            parts.extend(f"{item}|{pos}" for pos, item in enumerate(sequence))
            f.write(" ".join(parts) + "\n")


def prepare_single_domain(source_root, target_root, domain, out_name=None, len_max=50):
    out_name = out_name or DEFAULT_SINGLE_NAMES.get(domain, domain.lower())
    interactions = _load_inter(source_root, domain)
    users = sorted(interactions, key=_sort_key)
    items = sorted({item for seq in interactions.values() for item in seq}, key=_sort_key)
    item_map = {item: idx + 1 for idx, item in enumerate(items)}
    user_map = {user: idx for idx, user in enumerate(users)}

    sequences = []
    for user in users:
        mapped = [item_map[item] for item in interactions[user]][-len_max:]
        if mapped:
            sequences.append((user_map[user], mapped))

    out_dir = Path(target_root) / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "map_user.txt", user_map)
    _write_json(out_dir / "map_item.txt", {str(item): [idx, 0] for item, idx in item_map.items()})
    _write_sequences(out_dir / f"{out_name}_{len_max}_preprocessed.txt", sequences)

    return {"out_dir": str(out_dir), "n_user": len(sequences), "n_item": len(item_map)}


def _round_robin(seq_a, seq_b):
    merged = []
    max_len = max(len(seq_a), len(seq_b))
    for index in range(max_len):
        if index < len(seq_a):
            merged.append(seq_a[index])
        if index < len(seq_b):
            merged.append(seq_b[index])
    return merged


def prepare_pair(source_root, target_root, domain_a, domain_b, out_name=None, len_max=50, merge_strategy="round_robin"):
    out_name = out_name or DEFAULT_PAIR_NAMES.get((domain_a, domain_b), f"{domain_a.lower()}_{domain_b.lower()}")
    inter_a = _load_inter(source_root, domain_a)
    inter_b = _load_inter(source_root, domain_b)
    users = sorted(set(inter_a) & set(inter_b), key=_sort_key)

    items_a = sorted({item for user in users for item in inter_a[user]}, key=_sort_key)
    items_b = sorted({item for user in users for item in inter_b[user]}, key=_sort_key)
    map_a = {item: idx + 1 for idx, item in enumerate(items_a)}
    map_b = {item: len(map_a) + idx + 1 for idx, item in enumerate(items_b)}
    user_map = {user: idx for idx, user in enumerate(users)}

    sequences = []
    for user in users:
        seq_a = [map_a[item] for item in inter_a[user]]
        seq_b = [map_b[item] for item in inter_b[user]]
        merged = seq_a + seq_b if merge_strategy == "concat" else _round_robin(seq_a, seq_b)
        if merged:
            sequences.append((user_map[user], merged[-len_max:]))

    out_dir = Path(target_root) / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "map_user.txt", user_map)
    map_item = {}
    map_item.update({f"{domain_a}:{item}": [idx, 0] for item, idx in map_a.items()})
    map_item.update({f"{domain_b}:{item}": [idx, 1] for item, idx in map_b.items()})
    _write_json(out_dir / "map_item.txt", map_item)
    _write_sequences(out_dir / f"{out_name}_{len_max}_preprocessed.txt", sequences)

    return {"out_dir": str(out_dir), "n_user": len(sequences), "n_item_a": len(map_a), "n_item_b": len(map_b)}


def main():
    parser = argparse.ArgumentParser(description="Convert SDSR JSON data to this baseline project's sequence format.")
    subparsers = parser.add_subparsers(dest="mode", required=True)

    single = subparsers.add_parser("single")
    single.add_argument("--source-root", type=Path, required=True)
    single.add_argument("--target-root", type=Path, required=True)
    single.add_argument("--domain", required=True)
    single.add_argument("--out-name")
    single.add_argument("--len-max", type=int, default=50)

    pair = subparsers.add_parser("pair")
    pair.add_argument("--source-root", type=Path, required=True)
    pair.add_argument("--target-root", type=Path, required=True)
    pair.add_argument("--domain-a", required=True)
    pair.add_argument("--domain-b", required=True)
    pair.add_argument("--out-name")
    pair.add_argument("--len-max", type=int, default=50)
    pair.add_argument("--merge-strategy", choices=["round_robin", "concat"], default="round_robin")

    args = parser.parse_args()
    if args.mode == "single":
        stats = prepare_single_domain(args.source_root, args.target_root, args.domain, args.out_name, args.len_max)
    else:
        stats = prepare_pair(
            args.source_root,
            args.target_root,
            args.domain_a,
            args.domain_b,
            args.out_name,
            args.len_max,
            args.merge_strategy,
        )
    print(json.dumps(stats, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
