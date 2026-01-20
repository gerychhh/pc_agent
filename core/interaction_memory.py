from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT


MEMORY_PATH = PROJECT_ROOT / "interaction_memory.json"
MAX_HISTORY = 200


def _default_memory() -> dict[str, Any]:
    return {"routes": {}, "history": []}


def load_memory() -> dict[str, Any]:
    if not MEMORY_PATH.exists():
        return _default_memory()
    try:
        data = json.loads(MEMORY_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _default_memory()
    if not isinstance(data, dict):
        return _default_memory()
    if "routes" not in data or not isinstance(data.get("routes"), dict):
        data["routes"] = {}
    if "history" not in data or not isinstance(data.get("history"), list):
        data["history"] = []
    return data


def save_memory(memory: dict[str, Any]) -> None:
    tmp_path = Path(f"{MEMORY_PATH}.tmp")
    tmp_path.write_text(json.dumps(memory, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(MEMORY_PATH)


def get_route(query: str) -> str | None:
    if not query:
        return None
    memory = load_memory()
    route = memory.get("routes", {}).get(query)
    return route if isinstance(route, str) and route else None


def set_route(query: str, corrected_query: str) -> None:
    if not query or not corrected_query:
        return
    memory = load_memory()
    routes = memory.setdefault("routes", {})
    routes[query] = corrected_query
    save_memory(memory)


def record_history(query: str, response: str, resolved_query: str) -> None:
    if not query or not response:
        return
    memory = load_memory()
    history = memory.setdefault("history", [])
    history.append(
        {
            "query": query,
            "resolved_query": resolved_query,
            "response": response,
            "timestamp": int(time.time()),
        }
    )
    if len(history) > MAX_HISTORY:
        memory["history"] = history[-MAX_HISTORY:]
    save_memory(memory)
