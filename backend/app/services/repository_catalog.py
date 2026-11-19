"""Atomic summary metadata for fast repository listings."""

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from uuid import UUID

from backend.app.core.config import settings


def _summary(metadata: dict[str, Any]) -> dict[str, Any]:
    total_files = metadata.get("total_files")
    languages = metadata.get("language_distribution")
    if isinstance(total_files, bool) or not isinstance(total_files, int) or total_files < 0:
        raise ValueError("Repository metadata has an invalid total_files value")
    if not isinstance(languages, dict) or any(
        not isinstance(language, str)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 0
        for language, count in languages.items()
    ):
        raise ValueError("Repository metadata has an invalid language_distribution value")
    return {
        "total_files": total_files,
        "language_distribution": dict(sorted(languages.items())),
    }


def _metadata_path(repository_id: str) -> Path:
    UUID(repository_id)
    root = settings.DATA_DIR / "repository_metadata"
    if root.is_symlink():
        raise ValueError("Repository metadata directory cannot be a symbolic link")
    return root / f"{repository_id}.json"


def save_repository_summary(repository_id: str, metadata: dict[str, Any]) -> None:
    """Atomically persist only the listing fields, never source paths."""
    target = _metadata_path(repository_id)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        raise ValueError("Repository metadata file cannot be a symbolic link")
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{repository_id}.", suffix=".tmp", dir=target.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as output:
            json.dump(_summary(metadata), output, sort_keys=True)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, target)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def load_repository_summary(repository_id: str) -> dict[str, Any] | None:
    """Return validated listing metadata, or None for missing/malformed files."""
    try:
        path = _metadata_path(repository_id)
        if path.is_symlink():
            return None
        with path.open(encoding="utf-8") as source:
            value = json.load(source)
        return _summary(value) if isinstance(value, dict) else None
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None
