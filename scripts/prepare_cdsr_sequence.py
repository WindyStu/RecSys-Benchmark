from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def prepare_cdsr_sequence(
    source_dir: str | Path,
    target_root: str | Path,
    dataset: str,
    len_max: int = 50,
) -> dict[str, Any]:
    source = Path(source_dir)
    interactions = _read_mapping(source / f"{dataset}.inter.json")
    raw_item_map = _read_mapping(source / "map_item.txt")
    raw_user_map = _read_mapping(source / "map_user.txt")

    items_by_domain: dict[int, list[tuple[str, int]]] = {0: [], 1: []}
    for raw_id, value in raw_item_map.items():
        if not isinstance(value, list) or len(value) < 2:
            raise ValueError(f"Invalid item mapping for {raw_id}: expected [mapped_id, domain]")
        old_id, domain = int(value[0]), int(value[1])
        if domain not in items_by_domain:
            raise ValueError(f"Unsupported domain label {domain}; expected 0 or 1")
        items_by_domain[domain].append((str(raw_id), old_id))

    remapped_items: dict[str, list[int]] = {}
    old_to_new: dict[int, int] = {}
    next_id = 1
    for domain in (0, 1):
        for raw_id, old_id in sorted(items_by_domain[domain], key=lambda pair: pair[1]):
            if old_id in old_to_new:
                raise ValueError(f"Duplicate mapped item id: {old_id}")
            old_to_new[old_id] = next_id
            remapped_items[raw_id] = [next_id, domain]
            next_id += 1

    out_dir = Path(target_root) / dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_json(out_dir / "map_item.txt", remapped_items)
    _write_json(out_dir / f"map_item_{len_max}.txt", remapped_items)
    _write_json(out_dir / "map_user.txt", raw_user_map)
    _write_json(out_dir / f"map_user_{len_max}.txt", raw_user_map)

    lines = []
    for user_id in sorted(interactions, key=_sort_key):
        sequence = [old_to_new[int(item)] for item in interactions[user_id]][-len_max:]
        if sequence:
            tokens = " ".join(f"{item}|{position}" for position, item in enumerate(sequence))
            lines.append(f"{user_id} {tokens}")
    (out_dir / f"{dataset}_{len_max}_preprocessed.txt").write_text(
        "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
    )

    return {
        "out_dir": str(out_dir),
        "n_user": len(lines),
        "n_item_0": len(items_by_domain[0]),
        "n_item_1": len(items_by_domain[1]),
    }


def _read_mapping(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True), encoding="utf-8")


def _sort_key(value: str) -> tuple[int, int | str]:
    return (0, int(value)) if str(value).isdigit() else (1, str(value))


def main() -> None:
    parser = argparse.ArgumentParser(description="Bridge processed CDSR JSON into ABXI/MERIT sequence files.")
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--target-root", type=Path, required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--len-max", type=int, default=50)
    args = parser.parse_args()
    print(json.dumps(prepare_cdsr_sequence(args.source_dir, args.target_root, args.dataset, args.len_max), indent=2))


if __name__ == "__main__":
    main()
