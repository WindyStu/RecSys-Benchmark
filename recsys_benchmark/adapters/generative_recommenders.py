from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import torch

from recsys_benchmark.adapters.command import CommandAdapter


class GenerativeRecommendersAdapter(CommandAdapter):
    """Adapter for HSTU and related Generative Recommenders entrypoints."""

    def evaluate(self) -> dict[str, Any]:
        if self.config.get("prediction") or self.method.get("prediction"):
            return super().evaluate()
        native_metrics = self.method.get("native_metrics")
        if not isinstance(native_metrics, Mapping) or native_metrics.get("type") != "hstu_checkpoint":
            return super().evaluate()

        checkpoint_path = self._find_checkpoint(native_metrics)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        native = checkpoint.get("metrics", {})
        metrics: dict[str, float | str] = {
            "recall@5": float(native["hr@5"]),
            "recall@10": float(native["hr@10"]),
            "hitrate@5": float(native["hr@5"]),
            "hitrate@10": float(native["hr@10"]),
            "ndcg@5": float(native["ndcg@5"]),
            "ndcg@10": float(native["ndcg@10"]),
            "mrr@10": "N/A",
            "precision@5": "N/A",
            "precision@10": "N/A",
            "map@10": "N/A",
        }
        record = {
            "method_id": self.method.get("method_id", "hstu"),
            "dataset": self.dataset.get("dataset_id", "unknown"),
            "task": self.dataset.get("task", "sdsr"),
            "protocol": self.config.get("evaluation", {}).get("protocol", "full"),
            "seed": self.config.get("seed"),
            "eval_input_type": "native_metrics",
            "native_metrics_source": str(checkpoint_path),
            "metrics": metrics,
        }
        output_path = self.output_dir / "metrics.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        return {"status": "evaluated", "stage": "evaluate", "metrics_path": str(output_path), "checkpoint": str(checkpoint_path)}

    def _find_checkpoint(self, native_metrics: Mapping[str, Any]) -> Path:
        checkpoint_dir = Path(self._render(str(native_metrics["log_dir"])))
        pattern = self._render(str(native_metrics["pattern"]))
        matches = sorted(checkpoint_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
        if not matches:
            raise FileNotFoundError(f"No HSTU checkpoint found in {checkpoint_dir} with pattern {pattern}")
        return matches[0]
