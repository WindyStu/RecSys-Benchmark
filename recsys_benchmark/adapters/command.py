from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from recsys_benchmark.adapters.base import BaseAdapter


class CommandAdapter(BaseAdapter):
    """Adapter that runs original paper entrypoints through configured commands."""

    def prepare(self) -> dict[str, Any]:
        return self._run_stage("prepare")

    def train(self) -> dict[str, Any]:
        return self._run_stage("train")

    def predict(self) -> dict[str, Any]:
        return self._run_stage("predict")

    def evaluate(self) -> dict[str, Any]:
        return self._run_stage("evaluate")

    def _run_stage(self, stage: str) -> dict[str, Any]:
        command_templates = self.method.get("commands", {})
        if stage not in command_templates:
            return {"status": "skipped", "stage": stage, "reason": f"no {stage} command configured"}

        command = [self._render(str(part)) for part in command_templates[stage]]
        source = Path(str(self.method.get("source", ".")))
        dry_run = bool(self.config.get("dry_run", False))
        started = time.perf_counter()

        if dry_run:
            result = {"returncode": 0, "command": command, "mode": "dry_run", "stage": stage, "cwd": str(source)}
        else:
            completed = subprocess.run(command, cwd=source, check=False, text=True)
            result = {
                "returncode": completed.returncode,
                "command": command,
                "mode": "subprocess",
                "stage": stage,
                "cwd": str(source),
            }

        result["elapsed_seconds"] = time.perf_counter() - started
        self._write_stage_artifact(stage, result)
        return result

    def _render(self, value: str) -> str:
        context = {
            "seed": self.config.get("seed"),
            "output_dir": str(self.output_dir),
            "dataset": _to_namespace(self.dataset),
            "method": _to_namespace(self.method),
            "evaluation": _to_namespace(self.config.get("evaluation", {})),
        }
        return value.format_map(_SafeFormatMapping(context))

    def _write_stage_artifact(self, stage: str, result: Mapping[str, Any]) -> None:
        artifacts_dir = self.output_dir / "logs"
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        (artifacts_dir / f"{stage}.json").write_text(json.dumps(dict(result), indent=2), encoding="utf-8")


class _SafeFormatMapping(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _to_namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(**{str(key): _to_namespace(child) for key, child in value.items()})
    if isinstance(value, list):
        return [_to_namespace(child) for child in value]
    return value
