from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from .debug import debug_event


@dataclass
class RouteResult:
    type: str
    raw: str
    cmd_id: str | None = None
    params: dict[str, str] | None = None
    ask_text: str | None = None


class RouterLLM:
    def __init__(self, client: Any, model_name: str, allowlist: set[str]) -> None:
        self.client = client
        self.model_name = model_name
        self.allowlist = allowlist

    def route(self, text: str) -> RouteResult:
        prompt = (
            "[SYSTEM] Ты — маршрутизатор команд. Верни ровно одну строку.\n"
            "[FORMAT] COMMAND: CMD_ID (param=value, ...)\n"
            "[FORMAT] ASK: вопрос пользователю\n"
            "[RULES]\n"
            "- Команда должна быть из allowlist.\n"
            "- Если не хватает данных — ASK.\n"
            "- Не добавляй ничего кроме одной строки.\n"
            "- Примеры:\n"
            "  COMMAND: CMD_OPEN_URL (url=https://ru.wikipedia.org/)\n"
            "  COMMAND: CMD_OPEN_APP_SEARCH (app_name=dota 2)\n"
            "  ASK: Что открыть?\n"
            f"[ALLOWLIST] {', '.join(sorted(self.allowlist))}\n"
            f"[TEXT] {text}\n"
        )
        debug_event("ROUTER_PROMPT", prompt)
        try:
            response = self.client.chat(
                [{"role": "user", "content": prompt}],
                tools=[],
                model_name=self.model_name,
                tool_choice="none",
            )
            content = (response.choices[0].message.content or "").strip()
        except Exception as exc:
            return RouteResult(type="invalid", raw=f"error={exc}")
        return _parse_router_output(content, self.allowlist)


def _parse_router_output(content: str, allowlist: set[str]) -> RouteResult:
    raw = content.strip()
    if not raw:
        return RouteResult(type="invalid", raw=raw)
    if raw.upper().startswith("ASK:"):
        ask_text = raw.split(":", 1)[1].strip()
        return RouteResult(type="ask", raw=raw, ask_text=ask_text)
    if not raw.upper().startswith("COMMAND:"):
        return RouteResult(type="invalid", raw=raw)
    command_part = raw.split(":", 1)[1].strip()
    match = re.match(r"([A-Z0-9_]+)\s*(\((.*)\))?$", command_part)
    if not match:
        return RouteResult(type="invalid", raw=raw)
    cmd_id = match.group(1)
    if cmd_id not in allowlist:
        return RouteResult(type="invalid", raw=raw)
    params_raw = match.group(3) or ""
    params = parse_params(params_raw)
    return RouteResult(type="command", raw=raw, cmd_id=cmd_id, params=params)


def parse_params(params_text: str) -> dict[str, str]:
    cleaned = params_text.strip()
    if not cleaned:
        return {}
    pairs = [part.strip() for part in cleaned.split(",") if part.strip()]
    params: dict[str, str] = {}
    for pair in pairs:
        if "=" not in pair:
            continue
        key, value = pair.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'").strip()
        if key:
            params[key] = value
    return params
