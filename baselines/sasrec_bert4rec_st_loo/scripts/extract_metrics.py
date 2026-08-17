import argparse
import csv
import re
from pathlib import Path


FINAL_RE = re.compile(r"\[ Info \]\s+([^-]+)-([a-z]+)\s+\(")
DATA_RE = re.compile(r"\[Info\]\s+(\S+)\s+\(data:([^,\)]+)")


def _parse_values(line):
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    if len(cells) < 6 or cells[0] != "F":
        return []
    values = cells[1:]
    if len(values) == 5:
        return [("A/B", values)]
    if len(values) == 10:
        return [("A", values[:5]), ("B", values[5:])]
    if len(values) == 4:
        return [("A/B", [values[0], values[1], "", values[2], values[3]])]
    if len(values) == 8:
        return [
            ("A", [values[0], values[1], "", values[2], values[3]]),
            ("B", [values[4], values[5], "", values[6], values[7]]),
        ]
    return []


def extract_file(path):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    model = task = data = ""
    rows = []
    for line in lines:
        data_match = DATA_RE.search(line)
        if data_match:
            model, data = data_match.groups()

        final_match = FINAL_RE.search(line)
        if final_match:
            model, task = final_match.groups()

        parsed = _parse_values(line)
        if parsed:
            rows = [
                {
                    "log_file": str(path),
                    "model": model,
                    "task": task,
                    "data": data,
                    "setting": "F",
                    "domain": domain,
                    "hr5": values[0],
                    "hr10": values[1],
                    "ndcg5": values[2],
                    "ndcg10": values[3],
                    "mrr": values[4],
                }
                for domain, values in parsed
            ]
    return rows


def collect(log_dir, pattern):
    rows = []
    for path in sorted(Path(log_dir).glob(pattern)):
        rows.extend(extract_file(path))
    return rows


def write_csv(rows, out_path):
    fieldnames = ["log_file", "model", "task", "data", "setting", "domain", "hr5", "hr10", "ndcg5", "ndcg10", "mrr"]
    with Path(out_path).open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Extract final recommendation metrics from baseline logs.")
    parser.add_argument("--log-dir", type=Path, default=Path("log"))
    parser.add_argument("--pattern", default="*.log")
    parser.add_argument("--out", type=Path, default=Path("log") / "metrics_summary.csv")
    args = parser.parse_args()

    rows = collect(args.log_dir, args.pattern)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, args.out)
    print(f"Wrote {len(rows)} rows to {args.out}")


if __name__ == "__main__":
    main()
