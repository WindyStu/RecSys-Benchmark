from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

from recsys_benchmark.adapters.command import CommandAdapter


DUAL_ROW_RE = re.compile(r"^\s*\|" + r"\s*([0-9]+(?:\.[0-9]+)?)\s*\|" * 10 + r"\s*$")


class DualDomainLogAdapter(CommandAdapter):
    """Command adapter for rankers that report A/B HR, NDCG, and MRR rows."""

    def evaluate(self) -> dict[str, Any]:
        if self.config.get("prediction") or self.method.get("prediction"):
            return super().evaluate()
        native_metrics = self.method.get("native_metrics")
        if not isinstance(native_metrics, Mapping) or native_metrics.get("type") != "dual_domain_log":
            return super().evaluate()

        log_path = self._find_log(native_metrics)
        metrics = parse_dual_domain_log(log_path, self._domain_names())
        record = {
            "method_id": self.method.get("method_id", "unknown"),
            "dataset": self.dataset.get("dataset_id", "unknown"),
            "task": self.dataset.get("task", "cdsr"),
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

    def _find_log(self, native_metrics: Mapping[str, Any]) -> Path:
        log_dir = Path(self._render(str(native_metrics["log_dir"])))
        pattern = self._render(str(native_metrics["pattern"]))
        matches = sorted(log_dir.glob(pattern), key=lambda path: path.stat().st_mtime, reverse=True)
        if not matches:
            raise FileNotFoundError(f"No dual-domain log found in {log_dir} with pattern {pattern}")
        return matches[0]

    def _domain_names(self) -> tuple[str, str]:
        domains = self.dataset.get("domains", {})
        if isinstance(domains, Mapping):
            ordered = [value for _, value in sorted(domains.items(), key=lambda pair: int(pair[0]))]
        else:
            ordered = list(domains)
        if len(ordered) != 2:
            return ("0", "1")
        return (str(ordered[0]), str(ordered[1]))


def parse_dual_domain_log(path: str | Path, domain_names: Sequence[str]) -> dict[str, float | str]:
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        match = DUAL_ROW_RE.match(line)
        if match:
            rows.append([float(value) for value in match.groups()])
    if not rows:
        raise ValueError(f"No dual-domain final metric row found in: {path}")

    values = rows[-1]
    domain_a = dict(zip(("recall@5", "recall@10", "ndcg@5", "ndcg@10", "mrr@10"), values[:5]))
    domain_b = dict(zip(("recall@5", "recall@10", "ndcg@5", "ndcg@10", "mrr@10"), values[5:]))
    metrics: dict[str, float | str] = {}
    for name in domain_a:
        metrics[name] = (domain_a[name] + domain_b[name]) / 2
    metrics["hitrate@5"] = metrics["recall@5"]
    metrics["hitrate@10"] = metrics["recall@10"]
    for cutoff in (5, 10):
        for domain_name, domain_metrics in zip(domain_names, (domain_a, domain_b)):
            metrics[f"domainrecall@{cutoff}:{domain_name}"] = domain_metrics[f"recall@{cutoff}"]
            metrics[f"domainndcg@{cutoff}:{domain_name}"] = domain_metrics[f"ndcg@{cutoff}"]
        metrics[f"crossdomaintransfergap@{cutoff}"] = abs(
            domain_a[f"recall@{cutoff}"] - domain_b[f"recall@{cutoff}"]
        )
    metrics.update({"precision@5": "N/A", "precision@10": "N/A", "map@10": "N/A"})
    return metrics
