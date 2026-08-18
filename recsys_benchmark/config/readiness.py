from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def inspect_method_readiness(config_path: str | Path) -> dict[str, Any]:
    path = Path(config_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    missing = _missing_contract_fields(config)
    source_exists = Path(str(config.get("source", ""))).exists()
    if config.get("status") == "partial":
        computed_status = "partial"
    elif missing:
        computed_status = "source-integrated" if source_exists else "missing-source"
    else:
        computed_status = "adapter-ready"
    return {
        "method_id": config.get("method_id", path.stem),
        "declared_status": config.get("status", "unknown"),
        "computed_status": computed_status,
        "missing": missing,
        "source_exists": source_exists,
    }


def inspect_methods(config_dir: str | Path) -> list[dict[str, Any]]:
    return [inspect_method_readiness(path) for path in sorted(Path(config_dir).glob("*.yaml"))]


def _missing_contract_fields(config: dict[str, Any]) -> list[str]:
    missing = []
    for field in ("method_id", "method_type", "source", "adapter"):
        if not config.get(field):
            missing.append(field)
    if config.get("method_type") == "ranker":
        commands = config.get("commands") or {}
        if not any(stage in commands for stage in ("train", "predict")):
            missing.append("commands.train_or_predict")
        prediction = config.get("prediction") or {}
        native_metrics = config.get("native_metrics") or {}
        has_prediction = prediction.get("input_type") and prediction.get("path")
        has_native_metrics = native_metrics.get("type") and native_metrics.get("log_dir") and native_metrics.get("pattern")
        if not has_prediction and not has_native_metrics:
            missing.append("prediction.input_type")
            missing.append("prediction.path")
    return missing
