from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys_benchmark.data.sanitize import copy_sanitized_tree


def main() -> int:
    parser = argparse.ArgumentParser(description="Copy a third-party baseline tree while excluding generated/private files.")
    parser.add_argument("--source", required=True)
    parser.add_argument("--destination", required=True)
    parser.add_argument("--max-file-size-mb", type=int, default=8)
    args = parser.parse_args()
    stats = copy_sanitized_tree(args.source, args.destination, max_file_size_mb=args.max_file_size_mb)
    print(json.dumps(stats, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
