from __future__ import annotations

import json
from typing import Any

from .config import PROJECT_ROOT


STATE_PATH = PROJECT_ROOT / "state.json"
DEFAULT_STATE: dict[str, Any] = {
    "active_file": None,
    "active_url": None,
    "active_app": None,
    "recent_files": [],
    "recent_urls": [],
    "recent_apps": [],
}


def _normalize_state(state: dict[str, Any]) -> dict[str, Any]:
    normalized = DEFAULT_STATE.copy()
    for key in normalized:
        if key in state:
            normalized[key] = state[key]
    if not isinstance(normalized["recent_files"], list):
        normalized["recent_files"] = []
    if not isinstance(normalized["recent_urls"], list):
        normalized["recent_urls"] = []
    if not isinstance(normalized["recent_apps"], list):
        normalized["recent_apps"] = []
    return normalized


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return DEFAULT_STATE.copy()
    try:
        data = json.loads(STATE_PATH.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            return DEFAULT_STATE.copy()
        return _normalize_state(data)
    except (OSError, json.JSONDecodeError):
        return DEFAULT_STATE.copy()


def save_state(state: dict[str, Any]) -> None:
    normalized = _normalize_state(state)
    STATE_PATH.write_text(json.dumps(normalized, ensure_ascii=False, indent=2), encoding="utf-8")


def _unique_front(items: list[str], value: str, max_items: int) -> list[str]:
    filtered = [item for item in items if item != value]
    filtered.insert(0, value)
    return filtered[:max_items]


def set_active_file(path: str) -> None:
    state = load_state()
    state["active_file"] = path
    state["recent_files"] = _unique_front(state.get("recent_files", []), path, 20)
    save_state(state)


def get_active_file() -> str | None:
    state = load_state()
    active = state.get("active_file")
    return str(active) if active else None


def set_active_url(url: str) -> None:
    state = load_state()
    state["active_url"] = url
    state["recent_urls"] = _unique_front(state.get("recent_urls", []), url, 20)
    save_state(state)


def get_active_url() -> str | None:
    state = load_state()
    active = state.get("active_url")
    return str(active) if active else None


def set_active_app(app: str) -> None:
    state = load_state()
    state["active_app"] = app
    state["recent_apps"] = _unique_front(state.get("recent_apps", []), app, 20)
    save_state(state)


def get_active_app() -> str | None:
    state = load_state()
    active = state.get("active_app")
    return str(active) if active else None


def add_recent_file(path: str, max_items: int = 20) -> None:
    state = load_state()
    state["recent_files"] = _unique_front(state.get("recent_files", []), path, max_items)
    save_state(state)


def add_recent_url(url: str, max_items: int = 20) -> None:
    state = load_state()
    state["recent_urls"] = _unique_front(state.get("recent_urls", []), url, max_items)
    save_state(state)


def add_recent_app(app: str, max_items: int = 20) -> None:
    state = load_state()
    state["recent_apps"] = _unique_front(state.get("recent_apps", []), app, max_items)
    save_state(state)


def clear_state() -> None:
    save_state(DEFAULT_STATE.copy())
