from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from recsys_benchmark.adapters.command import CommandAdapter


class LetterAdapter(CommandAdapter):
    """Adapter for rankers and tokenizer components from the LETTER source tree."""

    def evaluate(self) -> dict[str, Any]:
        if self.config.get("prediction") or self.method.get("prediction"):
            return super().evaluate()
        native_metrics = self.method.get("native_metrics")
        if not isinstance(native_metrics, Mapping):
            return super().evaluate()
        metrics_type = native_metrics.get("type")
        if metrics_type not in {"cf_sasrec_best_json", "letter_result_json"}:
            return super().evaluate()

        source_path = self._find_native_result(native_metrics)
        native = json.loads(source_path.read_text(encoding="utf-8"))
        if metrics_type == "cf_sasrec_best_json":
            raw_metrics = native.get("test", {})
            hit5, hit10 = raw_metrics["HR@5"], raw_metrics["HR@10"]
            ndcg5, ndcg10 = raw_metrics["NDCG@5"], raw_metrics["NDCG@10"]
        else:
            raw_metrics = native.get("mean_results", {})
            hit5, hit10 = raw_metrics["hit@5"], raw_metrics["hit@10"]
            ndcg5, ndcg10 = raw_metrics["ndcg@5"], raw_metrics["ndcg@10"]
        metrics: dict[str, float | str] = {
            "recall@5": float(hit5),
            "recall@10": float(hit10),
            "hitrate@5": float(hit5),
            "hitrate@10": float(hit10),
            "ndcg@5": float(ndcg5),
            "ndcg@10": float(ndcg10),
            "mrr@10": "N/A",
            "precision@5": "N/A",
            "precision@10": "N/A",
            "map@10": "N/A",
        }
        record = {
            "method_id": self.method.get("method_id", "unknown"),
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

    def _find_native_result(self, native_metrics: Mapping[str, Any]) -> Path:
        result_dir = Path(self._render(str(native_metrics["log_dir"])))
        pattern = self._render(str(native_metrics["pattern"]))
        matches = sorted(result_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
        if not matches:
            raise FileNotFoundError(f"No LETTER native result found in {result_dir} with pattern {pattern}")
        return matches[0]
