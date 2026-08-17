from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from recsys_benchmark.aggregator.results import aggregate_runs, write_leaderboard


def main() -> int:
    parser = argparse.ArgumentParser(description="Aggregate RecSys-Benchmark metrics.json files.")
    parser.add_argument("--results", default="outputs/runs")
    parser.add_argument("--output-csv", default="results/leaderboard.csv")
    parser.add_argument("--output-md", default="results/leaderboard.md")
    args = parser.parse_args()
    rows = aggregate_runs(args.results)
    write_leaderboard(rows, args.output_csv, args.output_md)
    print(f"wrote {len(rows)} aggregate rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
