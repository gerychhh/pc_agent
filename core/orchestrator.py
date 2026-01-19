from __future__ import annotations

import json
import re
from typing import Any

from .llm_client import LLMClient
from .logger import SessionLogger
from .policy import RiskLevel, confirm_action, risk_level
from tools import desktop, filesystem, process


SYSTEM_PROMPT = (
    "Ты локальный ассистент управления ПК. "
    "Всегда используй инструменты для действий. "
    "Если пользователь просит открыть сервис (Яндекс Музыка, Telegram, Discord, Spotify и т.п.), "
    "сначала попытайся открыть установленное приложение (open_app / find_start_apps + open_start_app). "
    "Только если приложение не найдено или запуск не удался — предложи открыть веб-версию в браузере. "
    "URL открывай только через open_url. "
    "Если пользователь просит открыть сайт/страницу, используй open_url ОДИН раз. "
    "Если open_url вернул ok=true и done=true — сразу заверши задачу. "
    "Если ok=true — сразу заверши задачу и дай финальный ответ без дополнительных действий. "
    "Если действие вернуло ok=false — остановись и объясни почему, не повторяй тот же способ. "
    "Учитывай verified: если verified=false, объясни, что действие выполнено, но результат не подтвержден. "
    "Перед опасными действиями спрашивай подтверждение (инструменты сами спрашивают, но ты тоже предупреждай). "
    "Если действие не получилось: объясни причину и предложи шаги исправления. "
    "Никогда не повторяй одно и то же действие больше 2 раз; если не получается — остановись и объясни причину. "
    "После каждого действия смотри на tool result: если ok=false — не повторяй автоматически, а меняй стратегию. "
    "После каждого действия используй screenshot результаты, если они доступны."
)


TOOL_REGISTRY = {
    "screenshot": desktop.screenshot,
    "move_mouse": desktop.move_mouse,
    "click": desktop.click,
    "type_text": desktop.type_text,
    "press_key": desktop.press_key,
    "hotkey": desktop.hotkey,
    "locate_on_screen": desktop.locate_on_screen,
    "open_app": process.open_app,
    "open_url": process.open_url,
    "find_start_apps": process.find_start_apps,
    "find_start_menu_shortcuts": process.find_start_menu_shortcuts,
    "open_start_app": process.open_start_app,
    "run_cmd": process.run_cmd,
    "read_file": filesystem.read_file,
    "write_file": filesystem.write_file,
}


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "screenshot",
            "description": "Capture screenshot and save to screenshots directory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "note": {"type": "string"},
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "move_mouse",
            "description": "Move mouse to coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "click",
            "description": "Click at coordinates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "button": {"type": "string", "default": "left"},
                    "clicks": {"type": "integer", "default": 1},
                    "interval": {"type": "number", "default": 0.1},
                },
                "required": ["x", "y"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "type_text",
            "description": "Type text using keyboard.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                    "interval": {"type": "number", "default": 0.02},
                },
                "required": ["text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "press_key",
            "description": "Press a single key.",
            "parameters": {
                "type": "object",
                "properties": {"key": {"type": "string"}},
                "required": ["key"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "hotkey",
            "description": "Press multiple keys as a hotkey.",
            "parameters": {
                "type": "object",
                "properties": {"keys": {"type": "array", "items": {"type": "string"}}},
                "required": ["keys"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "locate_on_screen",
            "description": "Locate image on screen.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_path": {"type": "string"},
                    "confidence": {"type": "number", "default": 0.8},
                },
                "required": ["image_path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Open a Windows application.",
            "parameters": {
                "type": "object",
                "properties": {"app": {"type": "string"}},
                "required": ["app"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_url",
            "description": "Open a URL in the default browser.",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string"}},
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_start_apps",
            "description": "Search installed Start menu applications.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_start_menu_shortcuts",
            "description": "Search Start Menu .lnk shortcuts by name (useful when app isn't visible in Get-StartApps).",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string"},
                    "limit": {"type": "integer", "default": 10},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "open_start_app",
            "description": "Open a Start menu app by AppID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app_id": {"type": "string"},
                    "display_name": {"type": "string"},
                },
                "required": ["app_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_cmd",
            "description": "Run a shell command with safeguards.",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {"type": "string"},
                    "timeout_sec": {"type": "integer", "default": 15},
                },
                "required": ["cmd"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "max_chars": {"type": "integer", "default": 20000},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write text file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "content": {"type": "string"},
                },
                "required": ["path", "content"],
            },
        },
    },
]


class Orchestrator:
    def __init__(self) -> None:
        self.client = LLMClient()
        self.logger = SessionLogger()
        self.reset()

    def reset(self) -> None:
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    @staticmethod
    def _extract_single_url_intent(user_input: str) -> str | None:
        """Heuristic router for the common 'open a website' intent.

        We intentionally bypass the LLM for this intent to prevent repeated
        attempts using different strategies (browser open loops).
        """
        text = user_input.strip()
        lower = text.lower()

        # Must look like an 'open site/page' request.
        if not any(k in lower for k in ("открой", "открыть", "open")):
            return None
        if not any(k in lower for k in ("сайт", "страниц", "url", "http", "www", ".ru", ".com", ".net", ".org")):
            return None

        # Extract first URL-like token.
        m = re.search(r"(https?://[^\s]+)", text, re.IGNORECASE)
        if m:
            return m.group(1).rstrip('.,;')

        # Domain without scheme (e.g. music.yandex.ru)
        m2 = re.search(r"\b([a-z0-9\-]+(?:\.[a-z0-9\-]+)+)\b", text, re.IGNORECASE)
        if not m2:
            return None

        domain = m2.group(1).rstrip('.,;')

        # If the request is obviously more complex than "just open site", don't fast-path.
        # This keeps multi-step tasks under LLM control.
        extra_intent_words = ("введи", "набери", "найди", "зарегистр", "войти", "логин", "пароль", "скачай", "поиск")
        if any(w in lower for w in extra_intent_words):
            return None

        return "https://" + domain

    def run(self, user_input: str) -> str:
        # Fast-path: if the user explicitly asked to open a single URL/page,
        # do it deterministically once to avoid LLM "multiple strategy" loops.
        url = self._extract_single_url_intent(user_input)
        if url:
            # Ask for confirmation via policy
            level = risk_level("open_url", {"url": url})
            approved = confirm_action("open_url", {"url": url}, level)
            self.logger.log(
                "confirmation",
                {
                    "tool_name": "open_url",
                    "tool_args": {"url": url},
                    "risk": level.value,
                    "approved": approved,
                },
            )

            if not approved:
                assistant_content = "Ок, отменено пользователем."
                self.messages.append({"role": "assistant", "content": assistant_content})
                self.logger.log("assistant_response", {"content": assistant_content})
                return assistant_content

            tool_result = process.open_url(url)
            self.logger.log(
                "tool_call",
                {
                    "tool_name": "open_url",
                    "tool_args": {"url": url},
                    "tool_result": tool_result,
                },
            )

            # Provide tool result to the model and force a final response without more tools.
            self.messages.append({"role": "user", "content": user_input})
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": "direct_open_url",
                    "name": "open_url",
                    "content": tool_result,
                }
            )
            self.messages.append(
                {
                    "role": "system",
                    "content": "URL открыт (если ok=true). Дай финальный ответ и НЕ вызывай инструменты.",
                }
            )

            final_response = self.client.chat(self.messages, TOOLS_SCHEMA, tool_choice="none")
            assistant_content = final_response.choices[0].message.content or ""
            self.messages.append({"role": "assistant", "content": assistant_content})
            self.logger.log("assistant_response", {"content": assistant_content})
            return assistant_content

        self.messages.append({"role": "user", "content": user_input})
        self.logger.log("user_input", {"content": user_input})
        tool_repeat_counts: dict[str, int] = {}
        total_tool_calls = 0

        for _ in range(10):
            response = self.client.chat(self.messages, TOOLS_SCHEMA)
            message = response.choices[0].message

            if getattr(message, "tool_calls", None):
                for tool_call in message.tool_calls:
                    if total_tool_calls >= 8:
                        reason = "Превышен лимит инструментов (8). Пожалуйста, уточните задачу."
                        self.logger.log(
                            "loop_guard_triggered",
                            {
                                "reason": "tool_call_limit",
                                "limit": 8,
                                "total_tool_calls": total_tool_calls,
                            },
                        )
                        self.messages.append({"role": "assistant", "content": reason})
                        self.logger.log("assistant_response", {"content": reason})
                        return reason

                    tool_name = tool_call.function.name
                    args = json.loads(tool_call.function.arguments or "{}")
                    normalized_args = json.dumps(args, ensure_ascii=False, sort_keys=True)
                    repeat_key = f"{tool_name}:{normalized_args}"
                    repeats = tool_repeat_counts.get(repeat_key, 0) + 1
                    tool_repeat_counts[repeat_key] = repeats
                    if repeats > 2:
                        tool_result = json.dumps(
                            {
                                "ok": False,
                                "error": "loop_guard_triggered",
                                "details": {
                                    "tool_name": tool_name,
                                    "args": args,
                                    "repeats": repeats,
                                },
                            },
                            ensure_ascii=False,
                        )
                        self.logger.log(
                            "loop_guard_triggered",
                            {
                                "reason": "repeated_tool_call",
                                "tool_name": tool_name,
                                "tool_args": args,
                                "repeats": repeats,
                            },
                        )
                        self.messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tool_call.id,
                                "name": tool_name,
                                "content": tool_result,
                            }
                        )
                        assistant_content = (
                            f"Я зациклился на повторяющемся действии {tool_name}. "
                            "Останавливаюсь. Возможная причина: модель не видит изменений "
                            "на экране или не получает ожидаемый результат."
                        )
                        self.messages.append({"role": "assistant", "content": assistant_content})
                        self.logger.log("assistant_response", {"content": assistant_content})
                        return assistant_content

                    level = risk_level(tool_name, args)
                    approved = confirm_action(tool_name, args, level)
                    self.logger.log(
                        "confirmation",
                        {
                            "tool_name": tool_name,
                            "tool_args": args,
                            "risk": level.value,
                            "approved": approved,
                        },
                    )
                    if not approved:
                        tool_result = json.dumps(
                            {"ok": False, "error": "user_denied"},
                            ensure_ascii=False,
                        )
                    else:
                        tool_fn = TOOL_REGISTRY.get(tool_name)
                        if not tool_fn:
                            tool_result = json.dumps(
                                {"ok": False, "error": "tool_not_found"},
                                ensure_ascii=False,
                            )
                        else:
                            tool_result = tool_fn(**args)
                    self.logger.log(
                        "tool_call",
                        {
                            "tool_name": tool_name,
                            "tool_args": args,
                            "tool_result": tool_result,
                        },
                    )
                    self.messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_name,
                            "content": tool_result,
                        }
                    )
                    try:
                        parsed_result = json.loads(tool_result)
                    except json.JSONDecodeError:
                        parsed_result = {}
                    if "verified" in parsed_result:
                        self.logger.log(
                            "verification",
                            {
                                "tool_name": tool_name,
                                "verified": parsed_result.get("verified"),
                                "verify_reason": parsed_result.get("verify_reason"),
                            },
                        )
                    if parsed_result.get("ok") is False:
                        tool_repeat_counts[repeat_key] = 2
                    if parsed_result.get("ok") is False and parsed_result.get("error") == "app_not_found":
                        hint = parsed_result.get("user_hint") or "Не могу найти приложение."
                        self.messages.append({"role": "assistant", "content": hint})
                        self.logger.log("assistant_response", {"content": hint})
                        return hint
                    if parsed_result.get("ok") is False and parsed_result.get("error") == "use_open_url_tool":
                        self.messages.append(
                            {
                                "role": "system",
                                "content": "Запрос похож на URL. Используй open_url для открытия сайта.",
                            }
                        )
                        total_tool_calls += 1
                        continue
                    if parsed_result.get("ok") is False:
                        self.messages.append(
                            {
                                "role": "system",
                                "content": (
                                    "Действие не выполнено (ok=false). "
                                    "Дай финальный ответ и НЕ вызывай инструменты."
                                ),
                            }
                        )
                        final_response = self.client.chat(
                            self.messages,
                            TOOLS_SCHEMA,
                            tool_choice="none",
                        )
                        assistant_content = final_response.choices[0].message.content or ""
                        self.messages.append({"role": "assistant", "content": assistant_content})
                        self.logger.log("assistant_response", {"content": assistant_content})
                        return assistant_content
                    if parsed_result.get("ok") is True and parsed_result.get("done") is True:
                        self.messages.append(
                            {
                                "role": "system",
                                "content": (
                                    "Задача выполнена (done=true). Не вызывай больше tools. "
                                    "Дай финальный ответ пользователю."
                                ),
                            }
                        )
                        final_response = self.client.chat(
                            self.messages,
                            TOOLS_SCHEMA,
                            tool_choice="none",
                        )
                        assistant_content = final_response.choices[0].message.content or ""
                        self.messages.append({"role": "assistant", "content": assistant_content})
                        self.logger.log("assistant_response", {"content": assistant_content})
                        return assistant_content
                    total_tool_calls += 1
                continue

            assistant_content = message.content or ""
            self.messages.append({"role": "assistant", "content": assistant_content})
            self.logger.log("assistant_response", {"content": assistant_content})
            return assistant_content

        fallback = "Reached tool execution limit without a final response."
        self.logger.log("assistant_response", {"content": fallback})
        return fallback

    @staticmethod
    def _extract_single_url_intent(text: str) -> str | None:
        """Return URL if request looks like 'open this site/page' and nothing more."""
        t = (text or "").strip()
        if not t:
            return None

        # Only trigger for explicit "open website/page" intent in RU/EN.
        intent = re.search(r"\b(открой|открыть|open)\b", t, flags=re.IGNORECASE)
        if not intent:
            return None

        # Find first URL-like token
        m = re.search(r"(https?://\S+|www\.[^\s]+|[A-Za-z0-9.-]+\.[A-Za-z]{2,}[^\s]*)", t)
        if not m:
            return None

        # If user asks a complex task (and/then, type, click, search), don't short-circuit.
        complex_markers = ["и ", " затем", " потом", "напечат", "клик", "введ", "найди", "search", "type", "click"]
        lowered = t.lower()
        if any(marker in lowered for marker in complex_markers):
            return None

        url = m.group(1).strip().strip('"\'')
        if url.startswith("www."):
            url = "https://" + url
        elif not re.match(r"^https?://", url, flags=re.IGNORECASE):
            url = "https://" + url
        return url
