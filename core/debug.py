from __future__ import annotations

import logging
import os
from typing import Any

from .config import LOG_DIR


_LOGGER = logging.getLogger("pc_agent")
_LOGGER.setLevel(logging.INFO)
_LOGGER.propagate = False

if not _LOGGER.handlers:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(LOG_DIR / "pc_agent.log", encoding="utf-8")
    file_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    file_handler.setFormatter(file_formatter)
    _LOGGER.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_formatter = logging.Formatter("[%(levelname)s] %(message)s")
    console_handler.setFormatter(console_formatter)
    _LOGGER.addHandler(console_handler)


_DEBUG_ENABLED = os.getenv("PC_AGENT_DEBUG", "0") == "1"


def set_debug(enabled: bool) -> None:
    global _DEBUG_ENABLED
    _DEBUG_ENABLED = enabled
    os.environ["PC_AGENT_DEBUG"] = "1" if enabled else "0"
    _LOGGER.setLevel(logging.DEBUG if enabled else logging.INFO)


def debug_event(tag: str, message: str) -> None:
    if not _DEBUG_ENABLED:
        return
    _LOGGER.debug("[%s] %s", tag, message)


def info_event(tag: str, message: str) -> None:
    _LOGGER.info("[%s] %s", tag, message)


def truncate_text(text: str, limit: int = 400) -> str:
    if text is None:
        return ""
    if len(text) <= limit:
        return text
    return text[:limit] + "..."


def debug_context(tag: str, payload: Any, limit: int = 800) -> None:
    debug_event(tag, truncate_text(str(payload), limit))
