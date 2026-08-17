from __future__ import annotations

from abc import ABC
from pathlib import Path
from typing import Any, Mapping


class BaseAdapter(ABC):
    """Base class for wrapping heterogeneous paper codebases."""

    def __init__(self, config: Mapping[str, Any]):
        self.config = dict(config)
        self.method = dict(self.config.get("method", {}))
        self.dataset = dict(self.config.get("dataset", {}))
        self.output_dir = Path(str(self.config.get("output_dir", "outputs/runs/default")))

    def prepare(self) -> dict[str, Any]:
        return {"status": "skipped", "reason": "adapter has no prepare step"}

    def train(self) -> dict[str, Any]:
        return {"status": "skipped", "reason": "adapter has no train step"}

    def predict(self) -> dict[str, Any]:
        return {"status": "skipped", "reason": "adapter has no predict step"}

    def evaluate(self) -> dict[str, Any]:
        return {"status": "skipped", "reason": "adapter has no evaluate step"}

    def collect_artifacts(self) -> dict[str, Any]:
        return {"output_dir": str(self.output_dir)}
