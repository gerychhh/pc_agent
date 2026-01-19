from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any


CACHE_PATH = Path(__file__).resolve().parents[1] / "app_paths.json"


def normalize_app_name(name: str) -> str:
    return " ".join((name or "").strip().lower().split())


def load_cache() -> dict[str, Any]:
    if not CACHE_PATH.exists():
        return {}
    try:
        with CACHE_PATH.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, dict):
            return data
    except json.JSONDecodeError:
        return {}
    return {}


def save_cache(data: dict[str, Any]) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="app_paths_", suffix=".json", dir=str(CACHE_PATH.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            json.dump(data, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_path, CACHE_PATH)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def get_cached_launch(app_name: str) -> dict[str, Any] | None:
    normalized = normalize_app_name(app_name)
    if not normalized:
        return None
    data = load_cache()
    record = data.get(normalized)
    if isinstance(record, dict):
        return record
    return None


def update_cached_launch(app_name: str, record: dict[str, Any]) -> None:
    normalized = normalize_app_name(app_name)
    if not normalized:
        return
    data = load_cache()
    data[normalized] = record
    save_cache(data)


def invalidate_cached_launch(app_name: str) -> None:
    normalized = normalize_app_name(app_name)
    if not normalized:
        return
    data = load_cache()
    if normalized in data:
        data.pop(normalized, None)
        save_cache(data)
