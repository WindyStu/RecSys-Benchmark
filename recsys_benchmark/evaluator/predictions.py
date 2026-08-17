from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_COLUMNS = {
    "candidate_scores": {"user_id", "item_id", "score"},
    "topk": {"user_id", "item_id", "rank"},
}


def load_prediction_file(path: str | Path) -> list[dict[str, Any]]:
    prediction_path = Path(path)
    suffix = prediction_path.suffix.lower()
    if suffix == ".csv":
        return _load_csv(prediction_path)
    if suffix in {".jsonl", ".ndjson"}:
        return _load_jsonl(prediction_path)
    if suffix == ".json":
        data = json.loads(prediction_path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON prediction files must contain a list of rows")
        return [_coerce_row(row) for row in data]
    raise ValueError(f"Unsupported prediction file extension: {suffix}")


def validate_predictions(rows: list[dict[str, Any]], input_type: str) -> None:
    if input_type not in REQUIRED_COLUMNS:
        raise ValueError(f"Unsupported prediction input_type: {input_type}")
    required = REQUIRED_COLUMNS[input_type]
    for index, row in enumerate(rows):
        missing = sorted(column for column in required if column not in row or row[column] in {None, ""})
        if missing:
            raise ValueError(f"Prediction row {index} is missing required columns: {', '.join(missing)}")


def rows_to_recommendations(rows: list[dict[str, Any]], input_type: str) -> dict[str, list[str]]:
    validate_predictions(rows, input_type)
    if input_type == "topk":
        sorted_rows = sorted(rows, key=lambda row: (str(row["user_id"]), int(row["rank"])))
    else:
        sorted_rows = sorted(rows, key=lambda row: (str(row["user_id"]), -float(row["score"])))
    recommendations: dict[str, list[str]] = {}
    for row in sorted_rows:
        user_id = str(row["user_id"])
        recommendations.setdefault(user_id, []).append(str(row["item_id"]))
    return recommendations


def _load_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [_coerce_row(row) for row in csv.DictReader(handle)]


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                rows.append(_coerce_row(json.loads(stripped)))
    return rows


def _coerce_row(row: dict[str, Any]) -> dict[str, Any]:
    coerced = dict(row)
    if "rank" in coerced and coerced["rank"] not in {None, ""}:
        coerced["rank"] = int(coerced["rank"])
    if "score" in coerced and coerced["score"] not in {None, ""}:
        coerced["score"] = float(coerced["score"])
    return coerced
