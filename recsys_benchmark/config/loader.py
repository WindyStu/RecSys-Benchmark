from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path
from typing import Any, Mapping

import yaml


def load_experiment_config(
    experiment_path: str | Path,
    config_root: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    experiment_file = Path(experiment_path)
    root = Path(config_root) if config_root is not None else experiment_file.parents[1]
    experiment = _load_yaml(experiment_file)

    dataset_id = experiment.get("dataset")
    method_id = experiment.get("method")
    if not dataset_id:
        raise ValueError("Experiment config must define dataset")
    if not method_id:
        raise ValueError("Experiment config must define method")

    dataset = _load_yaml(root / "datasets" / f"{dataset_id}.yaml")
    dataset = _resolve_dataset_path(dataset, root)
    method = _load_yaml(root / "methods" / f"{method_id}.yaml")

    resolved = deepcopy(experiment)
    resolved["dataset"] = dataset
    resolved["method"] = method

    for key, value in (overrides or {}).items():
        _set_dotted(resolved, key, value)
    return resolved


def parse_overrides(values: list[str] | None) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Override must be key=value, got: {value}")
        key, raw = value.split("=", 1)
        overrides[key] = yaml.safe_load(raw)
    return overrides


def save_resolved_config(config: Mapping[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(dict(config), sort_keys=False, allow_unicode=True), encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    loaded = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(loaded, dict):
        raise ValueError(f"YAML file must contain a mapping: {path}")
    return loaded


def _resolve_dataset_path(dataset: dict[str, Any], config_root: Path) -> dict[str, Any]:
    resolved = dict(dataset)
    project_root = config_root.parent if config_root.name == "configs" else config_root
    if resolved.get("data_root_env"):
        env_name = str(resolved["data_root_env"])
        if env_name in os.environ:
            base = Path(os.environ[env_name])
            relative = resolved.get("relative_path")
            path = (base / str(relative)).resolve() if relative else base.resolve()
            resolved["path"] = str(path)
            resolved["path_root"] = str(path.parent)
        return resolved
    if resolved.get("data_root"):
        data_root = Path(str(resolved["data_root"]))
        if not data_root.is_absolute():
            data_root = project_root / data_root
        relative = resolved.get("relative_path")
        path = (data_root / str(relative)).resolve() if relative else data_root.resolve()
        resolved["path"] = str(path)
        resolved["path_root"] = str(path.parent)
    return resolved


def _set_dotted(config: dict[str, Any], dotted_key: str, value: Any) -> None:
    parts = dotted_key.split(".")
    cursor: dict[str, Any] = config
    for part in parts[:-1]:
        next_value = cursor.setdefault(part, {})
        if not isinstance(next_value, dict):
            raise ValueError(f"Cannot set nested override through non-mapping key: {part}")
        cursor = next_value
    cursor[parts[-1]] = value
