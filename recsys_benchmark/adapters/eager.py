from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from recsys_benchmark.adapters.command import CommandAdapter


class EAGERAdapter(CommandAdapter):
    """Adapter for the SDSR DIN + EAGER + full-ranking evaluation pipeline."""

    def evaluate(self) -> dict[str, Any]:
        if self.config.get("prediction") or self.method.get("prediction"):
            return super().evaluate()
        native_metrics = self.method.get("native_metrics")
        if not isinstance(native_metrics, Mapping) or native_metrics.get("type") != "eager_eval_json":
            return super().evaluate()

        source_path = self._find_result(native_metrics)
        native = json.loads(source_path.read_text(encoding="utf-8"))
        metrics: dict[str, float | str] = {
            name: float(native[name]) for name in ("recall@5", "recall@10", "ndcg@5", "ndcg@10")
        }
        metrics.update(
            {
                "hitrate@5": metrics["recall@5"],
                "hitrate@10": metrics["recall@10"],
                "mrr@10": "N/A",
                "precision@5": "N/A",
                "precision@10": "N/A",
                "map@10": "N/A",
            }
        )
        record = {
            "method_id": self.method.get("method_id", "eager"),
            "dataset": self.dataset.get("dataset_id", "unknown"),
            "task": self.dataset.get("task", "sdsr"),
            "protocol": self.config.get("evaluation", {}).get("protocol", "full"),
            "seed": self.config.get("seed"),
            "eval_input_type": "native_metrics",
            "native_metrics_source": str(source_path),
            "metrics": metrics,
        }
        output_path = self.output_dir / "metrics.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        return {"status": "evaluated", "stage": "evaluate", "metrics_path": str(output_path), "native_result": str(source_path)}

    def _find_result(self, native_metrics: Mapping[str, Any]) -> Path:
        result_dir = Path(self._render(str(native_metrics["log_dir"])))
        pattern = self._render(str(native_metrics["pattern"]))
        matches = sorted(result_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
        if not matches:
            raise FileNotFoundError(f"No EAGER evaluation result found in {result_dir} with pattern {pattern}")
        return matches[0]
