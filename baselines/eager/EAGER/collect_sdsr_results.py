import argparse
import csv
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Collect SDSR EAGER eval_results.json files into one CSV.")
    parser.add_argument("--output-root", default="runs/sdsr_outputs")
    parser.add_argument("--csv-path", default="")
    args = parser.parse_args()

    output_root = Path(args.output_root)
    csv_path = Path(args.csv_path) if args.csv_path else output_root / "summary.csv"
    rows = []
    for result_path in sorted(output_root.glob("*/eval_results.json")):
        with open(result_path, "r", encoding="utf-8") as f:
            rows.append(json.load(f))

    fieldnames = ["dataset", "recall@5", "recall@10", "ndcg@5", "ndcg@10", "test_users", "checkpoint"]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})
    print(f"wrote {csv_path}")


if __name__ == "__main__":
    main()
