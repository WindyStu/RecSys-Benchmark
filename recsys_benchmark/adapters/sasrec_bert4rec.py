from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from recsys_benchmark.adapters.command import CommandAdapter


class SASRecBERT4RecAdapter(CommandAdapter):
    """Adapter for the shared SASRec/BERT4Rec/STOSA single-domain baseline tree."""

    def evaluate(self) -> dict[str, Any]:
        if self.config.get("prediction") or self.method.get("prediction"):
            return super().evaluate()
        native_metrics = self.method.get("native_metrics")
        if isinstance(native_metrics, Mapping):
            return self._evaluate_native_metrics(native_metrics)
        return super().evaluate()

    def _evaluate_native_metrics(self, native_metrics: Mapping[str, Any]) -> dict[str, Any]:
        metrics_type = native_metrics.get("type")
        if metrics_type != "sasrec_st_log":
            raise ValueError(f"Unsupported native metrics type for SASRec-family adapter: {metrics_type}")

        log_path = self._find_native_log(native_metrics)
        metrics = parse_sasrec_st_log(log_path)
        record = {
            "method_id": self.method.get("method_id", "unknown"),
            "dataset": self.dataset.get("dataset_id", "unknown"),
            "task": self.dataset.get("task", self.config.get("task", "sdsr")),
            "protocol": self.config.get("evaluation", {}).get("protocol", "full"),
            "seed": self.config.get("seed"),
            "eval_input_type": "native_metrics",
            "native_metrics_source": str(log_path),
            "metrics": metrics,
        }
        output_path = self.output_dir / "metrics.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        return {"status": "evaluated", "stage": "evaluate", "metrics_path": str(output_path), "native_log": str(log_path)}

    def _find_native_log(self, native_metrics: Mapping[str, Any]) -> Path:
        log_dir = Path(self._render(str(native_metrics.get("log_dir", "{method.source}/log"))))
        pattern = self._render(str(native_metrics.get("pattern", "*.log")))
        matches = sorted(log_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
        if not matches:
            raise FileNotFoundError(f"No native SASRec-family log found in {log_dir} with pattern {pattern}")
        return matches[0]


FINAL_ROW_RE = re.compile(
    r"\|\s*F\s*\|\s*"
    r"(?P<hr5>[0-9.]+)\s*\|\s*"
    r"(?P<hr10>[0-9.]+)\s*\|\s*"
    r"(?P<ndcg5>[0-9.]+)\s*\|\s*"
    r"(?P<ndcg10>[0-9.]+)\s*\|\s*"
    r"(?P<mrr>[0-9.]+)\s*\|"
)


def parse_sasrec_st_log(path: str | Path) -> dict[str, float | str]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = FINAL_ROW_RE.search(line)
        if match:
            rows.append({key: float(value) for key, value in match.groupdict().items()})
    if not rows:
        raise ValueError(f"No final metric row found in native SASRec-family log: {path}")

    values = rows[-1]
    return {
        "recall@5": values["hr5"],
        "recall@10": values["hr10"],
        "hitrate@5": values["hr5"],
        "hitrate@10": values["hr10"],
        "ndcg@5": values["ndcg5"],
        "ndcg@10": values["ndcg10"],
        "mrr@10": values["mrr"],
        "precision@5": "N/A",
        "precision@10": "N/A",
        "map@10": "N/A",
    }
