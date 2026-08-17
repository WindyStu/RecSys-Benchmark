from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys_benchmark.data.sanitize import scan_for_secrets


def main() -> int:
    parser = argparse.ArgumentParser(description="Scan text files for likely API keys or secrets.")
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    findings = scan_for_secrets(args.paths)
    for path, line_number, line in findings:
        print(f"{path}:{line_number}: {line}")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())
