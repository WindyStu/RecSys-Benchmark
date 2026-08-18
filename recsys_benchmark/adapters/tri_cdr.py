from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping

from recsys_benchmark.adapters.command import CommandAdapter


class TriCDRAdapter(CommandAdapter):
    """Adapter for Tri-CDR's sampled target-domain evaluation output."""

    def evaluate(self) -> dict[str, Any]:
        if self.config.get("prediction") or self.method.get("prediction"):
            return super().evaluate()
        native_metrics = self.method.get("native_metrics")
        if not isinstance(native_metrics, Mapping) or native_metrics.get("type") != "tri_cdr_log":
            return super().evaluate()

        result_path = self._find_result(native_metrics)
        metrics = parse_tri_cdr_log(result_path)
        record = {
            "method_id": self.method.get("method_id", "unknown"),
            "dataset": self.dataset.get("dataset_id", "unknown"),
            "task": self.dataset.get("task", "cdsr"),
            "protocol": self.config.get("evaluation", {}).get("protocol", "sampled"),
            "seed": self.config.get("seed"),
            "eval_input_type": "native_metrics",
            "native_metrics_source": str(result_path),
            "metrics": metrics,
        }
        output_path = self.output_dir / "metrics.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
        return {
            "status": "evaluated",
            "stage": "evaluate",
            "metrics_path": str(output_path),
            "native_log": str(result_path),
        }

    def _find_result(self, native_metrics: Mapping[str, Any]) -> Path:
        result_dir = Path(self._render(str(native_metrics["log_dir"])))
        pattern = self._render(str(native_metrics["pattern"]))
        matches = sorted(result_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
        if not matches:
            raise FileNotFoundError(f"No Tri-CDR result found in {result_dir} with pattern {pattern}")
        return matches[0]


METRIC_RE = re.compile(r"(NDCG|HR)@(1|5|10|20|50):\s*([0-9.]+)|AUC:\s*([0-9.]+)")


def parse_tri_cdr_log(path: str | Path) -> dict[str, float | str]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        values: dict[str, float] = {}
        for match in METRIC_RE.finditer(line):
            if match.group(1):
                values[f"{match.group(1).lower()}@{match.group(2)}"] = float(match.group(3))
            else:
                values["auc"] = float(match.group(4))
        if {"ndcg@5", "ndcg@10", "hr@5", "hr@10", "auc"}.issubset(values):
            rows.append(values)
    if not rows:
        raise ValueError(f"No Tri-CDR metric row found in: {path}")

    best = max(rows, key=lambda row: row["ndcg@10"])
    return {
        "recall@5": best["hr@5"],
        "recall@10": best["hr@10"],
        "hitrate@5": best["hr@5"],
        "hitrate@10": best["hr@10"],
        "ndcg@5": best["ndcg@5"],
        "ndcg@10": best["ndcg@10"],
        "auc": best["auc"],
        "mrr@10": "N/A",
        "precision@5": "N/A",
        "precision@10": "N/A",
        "map@10": "N/A",
    }
