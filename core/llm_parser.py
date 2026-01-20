from __future__ import annotations

import json
import re
from typing import Any

from .command_engine import Action


CODE_BLOCK_RE = re.compile(r"```(python|powershell|json)\s*(.*?)```", re.DOTALL | re.IGNORECASE)


def parse_action_from_text(text: str) -> Action | None:
    if not text:
        return None
    match = CODE_BLOCK_RE.search(text)
    if match:
        lang = match.group(1).lower()
        script = match.group(2).strip()
        if lang in ("python", "powershell"):
            return Action(language=lang, script=script)

    tool_match = re.search(r"LANG:\s*(python|powershell)\s*\nSCRIPT:\s*(.+)", text, re.DOTALL | re.IGNORECASE)
    if tool_match:
        lang = tool_match.group(1).lower()
        script = tool_match.group(2).strip()
        return Action(language=lang, script=script)

    header_match = re.match(r"^(python|powershell)\s*\n(.*)", text.strip(), re.DOTALL | re.IGNORECASE)
    if header_match:
        lang = header_match.group(1).lower()
        script = header_match.group(2).strip()
        return Action(language=lang, script=script)

    inline_match = re.search(r"\b(powershell|python)\b\s*-Command\s+(.+)", text, re.IGNORECASE)
    if inline_match:
        lang = inline_match.group(1).lower()
        script = inline_match.group(2).strip().strip("\"")
        return Action(language=lang, script=script)

    return None


def parse_step_json(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    match = CODE_BLOCK_RE.search(text)
    if match and match.group(1).lower() == "json":
        candidate = match.group(2).strip()
        return _loads_json(candidate)

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        candidate = brace_match.group(0)
        return _loads_json(candidate)
    return None


def _loads_json(candidate: str) -> dict[str, Any] | None:
    try:
        data = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(data, dict):
        return data
    return None
