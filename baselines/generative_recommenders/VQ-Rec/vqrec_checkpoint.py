"""Checkpoint loading helpers for VQ-Rec fine-tuning."""

from contextlib import contextmanager
from typing import Any

import torch


def load_trusted_checkpoint(path: str, map_location: Any) -> dict:
    """Load a trusted VQ-Rec checkpoint across PyTorch versions.

    The bundled VQ-Rec checkpoints include a serialized RecBole Config object.
    PyTorch 2.6+ defaults ``torch.load`` to ``weights_only=True``, which rejects
    that legacy object. These checkpoints are local project artifacts, so we
    explicitly use the historical trusted-load behavior.
    """
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


@contextmanager
def trusted_torch_load_context():
    """Temporarily make default torch.load use trusted legacy checkpoint loading."""
    original_torch_load = torch.load

    def trusted_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        try:
            return original_torch_load(*args, **kwargs)
        except TypeError:
            kwargs.pop("weights_only", None)
            return original_torch_load(*args, **kwargs)

    torch.load = trusted_load
    try:
        yield
    finally:
        torch.load = original_torch_load


def checkpoint_dataset_name(checkpoint: dict) -> str:
    """Return the source dataset stored in a VQ-Rec checkpoint, if available."""
    config = checkpoint.get("config")
    if config is None:
        return "?"
    if isinstance(config, dict):
        return str(config.get("dataset", "?"))
    try:
        return str(config["dataset"])
    except (KeyError, TypeError, AttributeError):
        return str(getattr(config, "dataset", "?"))
