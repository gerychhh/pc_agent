from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


SEARCH_PATHS_FILE = Path(__file__).resolve().parents[1] / "app_search_paths.json"


def default_search_paths() -> list[str]:
    candidates = [
        os.environ.get("ProgramFiles"),
        os.environ.get("ProgramFiles(x86)"),
        os.environ.get("LOCALAPPDATA"),
        os.environ.get("APPDATA"),
        os.environ.get("ProgramData"),
    ]
    paths = [p for p in candidates if p and os.path.isdir(p)]
    return list(dict.fromkeys(paths))


def load_search_paths() -> list[str]:
    if not SEARCH_PATHS_FILE.exists():
        return []
    try:
        with SEARCH_PATHS_FILE.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        if isinstance(data, list):
            return [str(item) for item in data if isinstance(item, str)]
    except json.JSONDecodeError:
        return []
    return []


def save_search_paths(paths: list[str]) -> None:
    SEARCH_PATHS_FILE.parent.mkdir(parents=True, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix="app_search_paths_", suffix=".json", dir=str(SEARCH_PATHS_FILE.parent))
    try:
        with os.fdopen(tmp_fd, "w", encoding="utf-8") as handle:
            json.dump(paths, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(tmp_path, SEARCH_PATHS_FILE)
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def normalize_search_paths(paths: list[str]) -> list[str]:
    normalized: list[str] = []
    for path in paths:
        if not path:
            continue
        expanded = os.path.expandvars(path.strip())
        if expanded and os.path.isdir(expanded):
            normalized.append(expanded)
    return list(dict.fromkeys(normalized))
