from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


GROUP_KEYS = ("method_id", "dataset", "task", "protocol", "eval_input_type")


def aggregate_runs(runs_root: str | Path) -> list[dict[str, Any]]:
    run_metrics = _load_run_metrics(Path(runs_root))
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for record in run_metrics:
        key = tuple(record.get(field) for field in GROUP_KEYS)
        groups[key].append(record)

    rows = []
    for key, records in sorted(groups.items()):
        row = {field: value for field, value in zip(GROUP_KEYS, key)}
        row["num_runs"] = len(records)
        row["seeds"] = ",".join(str(record.get("seed")) for record in records)
        metric_names = sorted({name for record in records for name in record.get("metrics", {})})
        for metric_name in metric_names:
            values = [
                float(record["metrics"][metric_name])
                for record in records
                if isinstance(record.get("metrics", {}).get(metric_name), (int, float))
            ]
            if not values:
                row[f"{metric_name}_mean"] = "N/A"
                row[f"{metric_name}_std"] = "N/A"
                continue
            row[f"{metric_name}_mean"] = sum(values) / len(values)
            row[f"{metric_name}_std"] = _std(values)
        rows.append(row)
    return rows


def write_leaderboard(rows: Iterable[dict[str, Any]], output_csv: str | Path, output_md: str | Path | None = None) -> None:
    rows = list(rows)
    csv_path = Path(output_csv)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if output_md is not None:
        Path(output_md).write_text(_to_markdown(rows, fieldnames), encoding="utf-8")


def _load_run_metrics(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(root.rglob("metrics.json")):
        records.append(json.loads(path.read_text(encoding="utf-8")))
    return records


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def _to_markdown(rows: list[dict[str, Any]], fieldnames: list[str]) -> str:
    if not rows:
        return "| empty |\n| --- |\n"
    lines = [
        "| " + " | ".join(fieldnames) + " |",
        "| " + " | ".join("---" for _ in fieldnames) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fieldnames) + " |")
    return "\n".join(lines) + "\n"
