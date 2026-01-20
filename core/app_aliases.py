from __future__ import annotations

import json
from pathlib import Path

from .config import PROJECT_ROOT


ALIASES_PATH = PROJECT_ROOT / "app_aliases.json"


def load_aliases() -> dict[str, str]:
    if not ALIASES_PATH.exists():
        return {}
    try:
        data = json.loads(ALIASES_PATH.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(data, dict):
        return {}
    cleaned: dict[str, str] = {}
    for key, value in data.items():
        if isinstance(key, str) and isinstance(value, str):
            cleaned[key] = value
    return cleaned


def save_aliases(aliases: dict[str, str]) -> None:
    tmp_path = Path(f"{ALIASES_PATH}.tmp")
    tmp_path.write_text(json.dumps(aliases, ensure_ascii=False, indent=2), encoding="utf-8")
    tmp_path.replace(ALIASES_PATH)


def get_alias(key: str) -> str | None:
    if not key:
        return None
    aliases = load_aliases()
    return aliases.get(key)


def set_alias(key: str, value: str) -> None:
    if not key or not value:
        return
    aliases = load_aliases()
    aliases[key] = value
    save_aliases(aliases)
