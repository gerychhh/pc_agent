from __future__ import annotations

import json
from typing import Any

from .llm_client import LLMClient
from .logger import SessionLogger
from .policy import RiskLevel, confirm_action, risk_level
from tools import desktop, filesystem, process


SYSTEM_PROMPT = (
    "Ты локальный ассистент управления ПК. "
    "Всегда используй инструменты для действий. "
    "Перед опасными действиями спрашивай подтверждение (инструменты сами спрашивают, но ты тоже предупреждай). "
    "Если действие не получилось: объясни причину и предложи шаги исправления. "
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

    def run(self, user_input: str) -> str:
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
                    total_tool_calls += 1
                continue

            assistant_content = message.content or ""
            self.messages.append({"role": "assistant", "content": assistant_content})
            self.logger.log("assistant_response", {"content": assistant_content})
            return assistant_content

        fallback = "Reached tool execution limit without a final response."
        self.logger.log("assistant_response", {"content": fallback})
        return fallback
