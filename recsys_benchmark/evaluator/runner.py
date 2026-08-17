from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from recsys_benchmark.evaluator.metrics import evaluate_recommendations
from recsys_benchmark.evaluator.predictions import load_prediction_file, rows_to_recommendations


def run_evaluation(
    predictions_path: str | Path,
    ground_truth_path: str | Path,
    output_path: str | Path,
    input_type: str,
    cutoffs: Iterable[int],
    metadata: Mapping[str, Any],
    catalog_path: str | Path | None = None,
    item_metadata_path: str | Path | None = None,
) -> dict[str, Any]:
    rows = load_prediction_file(predictions_path)
    recommendations = rows_to_recommendations(rows, input_type=input_type)
    ground_truth = load_ground_truth(ground_truth_path)
    catalog_items = load_catalog(catalog_path) if catalog_path else None
    item_domains = load_item_domains(item_metadata_path) if item_metadata_path else None

    metrics = evaluate_recommendations(
        recommendations=recommendations,
        ground_truth=ground_truth,
        cutoffs=list(cutoffs),
        catalog_items=catalog_items,
        item_domains=item_domains,
    )
    record = dict(metadata)
    record["metrics"] = metrics

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return record


def load_ground_truth(path: str | Path) -> dict[str, set[str]]:
    ground_truth_path = Path(path)
    if ground_truth_path.suffix.lower() == ".csv":
        with ground_truth_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            truth: dict[str, set[str]] = {}
            for row in reader:
                truth.setdefault(str(row["user_id"]), set()).add(str(row["item_id"]))
            return truth
    if ground_truth_path.suffix.lower() in {".jsonl", ".ndjson"}:
        truth = {}
        with ground_truth_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                truth.setdefault(str(row["user_id"]), set()).add(str(row["item_id"]))
            return truth
    raise ValueError(f"Unsupported ground truth extension: {ground_truth_path.suffix}")


def load_catalog(path: str | Path) -> set[str]:
    catalog_path = Path(path)
    if catalog_path.suffix.lower() == ".csv":
        with catalog_path.open("r", encoding="utf-8", newline="") as handle:
            return {str(row["item_id"]) for row in csv.DictReader(handle)}
    if catalog_path.suffix.lower() == ".json":
        data = json.loads(catalog_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {str(item_id) for item_id in data}
        return {str(row["item_id"]) for row in data}
    raise ValueError(f"Unsupported catalog extension: {catalog_path.suffix}")


def load_item_domains(path: str | Path) -> dict[str, str]:
    metadata_path = Path(path)
    if metadata_path.suffix.lower() == ".csv":
        with metadata_path.open("r", encoding="utf-8", newline="") as handle:
            return {str(row["item_id"]): str(row["domain"]) for row in csv.DictReader(handle) if row.get("domain")}
    if metadata_path.suffix.lower() == ".json":
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return {
                str(item_id): str(value["domain"] if isinstance(value, dict) else value)
                for item_id, value in data.items()
                if (isinstance(value, dict) and "domain" in value) or not isinstance(value, dict)
            }
        return {str(row["item_id"]): str(row["domain"]) for row in data if "domain" in row}
    raise ValueError(f"Unsupported item metadata extension: {metadata_path.suffix}")
