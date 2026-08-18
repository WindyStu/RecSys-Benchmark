from __future__ import annotations

import re
import shutil
from pathlib import Path
from typing import Iterable


EXCLUDED_DIR_NAMES = {
    ".claude", ".git", ".idea", ".ipynb_checkpoints", "__pycache__", "checkpoint", "checkpoints",
    "ckpt", "exps", "log", "logs", "lsf_logs", "output", "outputs", "pretrained", "runs", "tmp", "wandb",
}

EXCLUDED_FILE_SUFFIXES = {
    ".7z", ".ckpt", ".csv", ".err", ".gz", ".index", ".joblib", ".jsonl", ".log", ".npy", ".npz",
    ".parquet", ".pkl", ".pt", ".pth", ".tar", ".tgz", ".zip",
}

TEXT_SUFFIXES = {
    ".bat", ".cfg", ".gin", ".ini", ".json", ".md", ".py", ".sh", ".toml", ".txt", ".yaml", ".yml",
}

SECRET_PATTERNS = [
    re.compile(r"sk-[A-Za-z0-9._-]{8,}"),
    re.compile(r"(?i)(api[_-]?key\s*=\s*)['\"][^'\"]+['\"]"),
    re.compile(r"(?i)(access[_-]?key\s*=\s*)['\"][^'\"]+['\"]"),
    re.compile(r"(?i)(secret\s*=\s*)['\"][^'\"]+['\"]"),
]

BASE_URL_PATTERNS = [
    re.compile(r"(?i)(base[_-]?url\s*=\s*)['\"]https?://[^'\"]+['\"]"),
    re.compile(r"(?i)(openai_base_url\s*:\s*)https?://\S+"),
]


def should_exclude_path(relative_path: Path) -> bool:
    parts = {part.lower() for part in relative_path.parts}
    if parts & EXCLUDED_DIR_NAMES:
        if relative_path.suffix.lower() == ".py":
            return False
        return True
    if "superpowers" in parts:
        return True
    if parts & {"data", "dataset", "datasets"} and relative_path.suffix.lower() not in {
        ".py", ".md", ".yaml", ".yml", ".sh"
    }:
        return True
    name = relative_path.name.lower()
    if name in {"workspace.xml", ".ds_store", "map_item.txt", "map_user.txt"}:
        return True
    if any(name.endswith(suffix) for suffix in (".inter.json", ".item.json", ".index.json", ".emb.json")):
        return True
    if relative_path.suffix.lower() in EXCLUDED_FILE_SUFFIXES:
        return True
    return False


def copy_sanitized_tree(source: str | Path, destination: str | Path, max_file_size_mb: int = 8) -> dict[str, int]:
    source_path = Path(source)
    destination_path = Path(destination)
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    destination_path.mkdir(parents=True, exist_ok=True)

    stats = {"copied_files": 0, "excluded_files": 0, "redacted_files": 0}
    max_bytes = max_file_size_mb * 1024 * 1024
    for file_path in source_path.rglob("*"):
        if not file_path.is_file():
            continue
        relative = file_path.relative_to(source_path)
        if should_exclude_path(relative) or file_path.stat().st_size > max_bytes:
            stats["excluded_files"] += 1
            continue
        target = destination_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if file_path.suffix.lower() in TEXT_SUFFIXES:
            text = file_path.read_text(encoding="utf-8", errors="ignore")
            redacted = redact_secrets(text)
            target.write_text(redacted, encoding="utf-8")
            if redacted != text:
                stats["redacted_files"] += 1
        else:
            shutil.copy2(file_path, target)
        stats["copied_files"] += 1
    return stats


def redact_secrets(text: str) -> str:
    redacted = text
    for pattern in SECRET_PATTERNS:
        redacted = pattern.sub(_redact_match, redacted)
    for pattern in BASE_URL_PATTERNS:
        redacted = pattern.sub(_redact_base_url_match, redacted)
    return redacted


def scan_for_secrets(paths: Iterable[str | Path]) -> list[tuple[str, int, str]]:
    findings = []
    for path_value in paths:
        path = Path(path_value)
        files = [path] if path.is_file() else [child for child in path.rglob("*") if child.is_file()]
        for file_path in files:
            if file_path.suffix.lower() not in TEXT_SUFFIXES:
                continue
            for line_number, line in enumerate(
                file_path.read_text(encoding="utf-8", errors="ignore").splitlines(), start=1
            ):
                if _is_allowed_placeholder_line(line):
                    continue
                if any(pattern.search(line) for pattern in SECRET_PATTERNS):
                    findings.append((str(file_path), line_number, line.strip()))
    return findings


def _redact_match(match: re.Match[str]) -> str:
    text = match.group(0)
    if "=" in text:
        prefix = text.split("=", 1)[0]
        return f'{prefix}= "REPLACE_WITH_ENV_VAR"'
    return "REPLACE_WITH_ENV_VAR"


def _redact_base_url_match(match: re.Match[str]) -> str:
    prefix = match.group(1)
    if "=" in prefix:
        return f'{prefix}"REPLACE_WITH_BASE_URL"'
    return f"{prefix}REPLACE_WITH_BASE_URL"


def _is_allowed_placeholder_line(line: str) -> bool:
    return any(marker in line for marker in ("REPLACE_WITH_ENV_VAR", '"..."', "'...'", "sk-test"))
