from __future__ import annotations

import json
import shutil
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping, Sequence

from recsys_benchmark.adapters.base import BaseAdapter
from recsys_benchmark.evaluator.runner import run_evaluation


class CommandAdapter(BaseAdapter):
    """Adapter that runs original paper entrypoints through configured commands."""

    def prepare(self) -> dict[str, Any]:
        binding_result = self._materialize_data_bindings()
        command_result = self._run_stage("prepare")
        if command_result.get("status") == "skipped" and binding_result["bindings"]:
            return {"status": "prepared", "stage": "prepare", **binding_result}
        if binding_result["bindings"]:
            command_result["data_bindings"] = binding_result["bindings"]
        return command_result

    def train(self) -> dict[str, Any]:
        return self._run_stage("train")

    def predict(self) -> dict[str, Any]:
        return self._run_stage("predict")

    def evaluate(self) -> dict[str, Any]:
        prediction = self.config.get("prediction") or self.method.get("prediction")
        if isinstance(prediction, Mapping):
            return self._run_unified_evaluation(prediction)
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
            result = {
                "returncode": 0,
                "command": command,
                "mode": "dry_run",
                "stage": stage,
                "cwd": str(source),
                "stdout": "",
                "stderr": "",
            }
        else:
            completed = subprocess.run(command, cwd=source, check=False, text=True, capture_output=True)
            result = {
                "returncode": completed.returncode,
                "command": command,
                "mode": "subprocess",
                "stage": stage,
                "cwd": str(source),
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }

        result["elapsed_seconds"] = time.perf_counter() - started
        self._write_stage_artifact(stage, result)
        if int(result.get("returncode", 0)) != 0:
            raise RuntimeError(f"{stage} failed with return code {result['returncode']}: {' '.join(command)}")
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

    def _materialize_data_bindings(self) -> dict[str, Any]:
        bindings = []
        for binding in self.method.get("data_bindings", []):
            source = Path(self._render(str(binding["from"])))
            target = Path(self._render(str(binding["to"])))
            mode = str(binding.get("mode", "copy"))
            if not source.exists():
                raise FileNotFoundError(f"data binding source does not exist: {source}")
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists():
                bindings.append({"from": str(source), "to": str(target), "mode": mode, "status": "exists"})
                continue
            if mode == "copy":
                if source.is_dir():
                    shutil.copytree(source, target)
                else:
                    shutil.copy2(source, target)
            elif mode == "symlink":
                target.symlink_to(source, target_is_directory=source.is_dir())
            else:
                raise ValueError(f"Unsupported data binding mode: {mode}")
            bindings.append({"from": str(source), "to": str(target), "mode": mode, "status": "created"})
        return {"bindings": bindings}

    def _run_unified_evaluation(self, prediction: Mapping[str, Any]) -> dict[str, Any]:
        output_path = self.output_dir / "metrics.json"
        evaluation = self.config.get("evaluation", {})
        record = run_evaluation(
            predictions_path=self._render(str(prediction["path"])),
            ground_truth_path=self._render(str(prediction["ground_truth"])),
            output_path=output_path,
            input_type=str(prediction["input_type"]),
            cutoffs=evaluation.get("cutoffs", [5, 10]),
            metadata={
                "method_id": self.method.get("method_id", "unknown"),
                "dataset": self.dataset.get("dataset_id", "unknown"),
                "task": self.dataset.get("task", self.config.get("task", "unknown")),
                "protocol": evaluation.get("protocol", "full"),
                "seed": self.config.get("seed"),
                "eval_input_type": prediction["input_type"],
            },
            catalog_path=self._render(str(prediction["catalog"])) if prediction.get("catalog") else None,
            item_metadata_path=self._render(str(prediction["item_metadata"])) if prediction.get("item_metadata") else None,
        )
        return {"status": "evaluated", "stage": "evaluate", "metrics_path": str(output_path), "metrics": record["metrics"]}


class _SafeFormatMapping(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


def _to_namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(**{str(key): _to_namespace(child) for key, child in value.items()})
    if isinstance(value, list):
        return [_to_namespace(child) for child in value]
    return value
