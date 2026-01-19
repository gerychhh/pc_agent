from __future__ import annotations

import json
import re
from typing import Any

from .llm_client import LLMClient
from .logger import SessionLogger
from .policy import confirm_action, risk_level, risk_reason
from tools import commands, filesystem, process


SYSTEM_PROMPT = (
    "Ты локальный ассистент управления ПК. "
    "НИКОГДА не используй управление мышью, клики или ввод с клавиатуры. "
    "Все действия выполняй только через инструменты open_app/open_url/run_powershell/run_cmd/run_python_script. "
    "Для создания txt/docx используй специализированные инструменты write_text_file_lines/create_docx, "
    "а не run_python_script. "
    "Перед созданием файлов на рабочем столе сначала вызывай get_known_paths, "
    "чтобы получить корректный путь Desktop. "
    "Если нужен вывод о результате — используй поля verified/verify_reason. "
    "Если пользователь дал точный путь к .exe, используй open_app с этим путем, не заменяй на другое имя. "
    "Если пользователь просит открыть сервис (Яндекс Музыка, Telegram, Discord, Spotify и т.п.), "
    "сначала попытайся открыть установленное приложение через open_app. "
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
    "После каждого действия смотри на tool result: если ok=false — не повторяй автоматически, а меняй стратегию."
)


TOOL_REGISTRY = {
    "open_app": process.open_app,
    "open_url": process.open_url,
    "run_powershell": commands.run_powershell,
    "run_cmd": commands.run_cmd,
    "run_python_script": commands.run_python_script,
    "get_known_paths": filesystem.get_known_paths,
    "write_text_file_lines": filesystem.write_text_file_lines,
    "create_docx": filesystem.create_docx,
    "read_file": filesystem.read_file,
    "write_file": filesystem.write_file,
}


TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "open_app",
            "description": "Open a Windows application.",
            "parameters": {
                "type": "object",
                "properties": {
                    "app": {"type": "string"},
                    "alias": {"type": "string"},
                },
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
            "name": "run_powershell",
            "description": "Run a PowerShell command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_sec": {"type": "integer", "default": 20},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_cmd",
            "description": "Run a CMD command.",
            "parameters": {
                "type": "object",
                "properties": {
                    "command": {"type": "string"},
                    "timeout_sec": {"type": "integer", "default": 20},
                },
                "required": ["command"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python_script",
            "description": "Run a short Python script via file execution.",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "timeout_sec": {"type": "integer", "default": 20},
                },
                "required": ["code"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_known_paths",
            "description": "Return common user paths (Desktop/Documents/Downloads/Home).",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_text_file_lines",
            "description": "Write repeated text lines to a file.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "line_template": {"type": "string"},
                    "count": {"type": "integer"},
                    "add_newline": {"type": "boolean", "default": True},
                },
                "required": ["path", "line_template", "count"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "create_docx",
            "description": "Create a .docx document with optional title and paragraphs.",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string"},
                    "title": {"type": "string"},
                    "paragraphs": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["path", "paragraphs"],
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
        self.pending_app_name: str | None = None
        self._no_arg_tools = {"get_known_paths"}
        self.reset()

    def reset(self) -> None:
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.pending_app_name = None

    @staticmethod
    def _extract_single_url_intent(user_input: str) -> str | None:
        """Heuristic router for the common 'open a website' intent."""
        text = user_input.strip()
        lower = text.lower()

        if not any(k in lower for k in ("открой", "открыть", "open")):
            return None
        if not any(k in lower for k in ("сайт", "страниц", "url", "http", "www", ".ru", ".com", ".net", ".org")):
            return None

        m = re.search(r"(https?://[^\s]+)", text, re.IGNORECASE)
        if m:
            return m.group(1).rstrip(".,;")

        m2 = re.search(r"\b([a-z0-9\-]+(?:\.[a-z0-9\-]+)+)\b", text, re.IGNORECASE)
        if not m2:
            return None

        domain = m2.group(1).rstrip(".,;")
        extra_intent_words = ("введи", "набери", "найди", "зарегистр", "войти", "логин", "пароль", "скачай", "поиск")
        if any(w in lower for w in extra_intent_words):
            return None

        return "https://" + domain

    @staticmethod
    def _extract_exe_path(user_input: str) -> str | None:
        if not user_input:
            return None
        match = re.search(r"([A-Za-z]:\\\\[^\"'\n]+?\\.exe)", user_input)
        if not match:
            return None
        return match.group(1)

    def _log_tool_debug(self, tool_name: str, args: dict[str, Any], tool_result: str) -> None:
        try:
            parsed = json.loads(tool_result)
        except json.JSONDecodeError:
            parsed = {}
        exec_cmd = None
        details = parsed.get("details") if isinstance(parsed, dict) else None
        if isinstance(details, dict):
            exec_cmd = details.get("exec")
        self.logger.log_tool_run(
            tool_name,
            args,
            exec_cmd,
            parsed.get("duration_ms") if isinstance(parsed, dict) else None,
            parsed.get("stdout") if isinstance(parsed, dict) else None,
            parsed.get("stderr") if isinstance(parsed, dict) else None,
            parsed.get("ok") if isinstance(parsed, dict) else None,
            parsed.get("verified") if isinstance(parsed, dict) else None,
        )

    @staticmethod
    def _tool_calls_summary(tool_calls: list[Any]) -> list[dict[str, Any]]:
        summary: list[dict[str, Any]] = []
        for call in tool_calls:
            name = call.function.name
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {"_raw": call.function.arguments}
            summary.append({"tool": name, "args": args})
        return summary

    def run(self, user_input: str) -> str:
        self.logger.log_user_input(user_input, len(self.messages))
        self.logger.log("user_input", {"content": user_input})

        exe_path = self._extract_exe_path(user_input)
        if exe_path:
            app_alias = self.pending_app_name
            level = risk_level("open_app", {"app": exe_path})
            reason = risk_reason("open_app", {"app": exe_path}, level)
            approved = confirm_action("open_app", {"app": exe_path}, level)
            self.logger.log_policy(level.value, reason, approved)
            self.logger.log(
                "confirmation",
                {
                    "tool_name": "open_app",
                    "tool_args": {"app": exe_path, "alias": app_alias},
                    "risk": level.value,
                    "risk_reason": reason,
                    "approved": approved,
                },
            )

            if not approved:
                assistant_content = "Ок, отменено пользователем."
                self.messages.append({"role": "assistant", "content": assistant_content})
                self.logger.log("assistant_response", {"content": assistant_content})
                self.logger.log_final(assistant_content)
                return assistant_content

            tool_result = process.open_app(exe_path, alias=app_alias)
            self._log_tool_debug("open_app", {"app": exe_path, "alias": app_alias}, tool_result)
            self.logger.log(
                "tool_call",
                {
                    "tool_name": "open_app",
                    "tool_args": {"app": exe_path, "alias": app_alias},
                    "tool_result": tool_result,
                },
            )

            self.messages.append({"role": "user", "content": user_input})
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": "direct_open_app",
                    "name": "open_app",
                    "content": tool_result,
                }
            )
            self.messages.append(
                {
                    "role": "system",
                    "content": "Приложение открыто (если ok=true). Дай финальный ответ и НЕ вызывай инструменты.",
                }
            )

            final_response = self.client.chat(self.messages, TOOLS_SCHEMA, tool_choice="none")
            assistant_content = final_response.choices[0].message.content or ""
            self.logger.log_llm_response(
                assistant_content,
                self._tool_calls_summary(getattr(final_response.choices[0].message, "tool_calls", []) or []),
            )
            self.messages.append({"role": "assistant", "content": assistant_content})
            self.logger.log("assistant_response", {"content": assistant_content})
            self.logger.log_final(assistant_content)
            self.pending_app_name = None
            return assistant_content

        url = self._extract_single_url_intent(user_input)
        if url:
            level = risk_level("open_url", {"url": url})
            reason = risk_reason("open_url", {"url": url}, level)
            approved = confirm_action("open_url", {"url": url}, level)
            self.logger.log_policy(level.value, reason, approved)
            self.logger.log(
                "confirmation",
                {
                    "tool_name": "open_url",
                    "tool_args": {"url": url},
                    "risk": level.value,
                    "risk_reason": reason,
                    "approved": approved,
                },
            )

            if not approved:
                assistant_content = "Ок, отменено пользователем."
                self.messages.append({"role": "assistant", "content": assistant_content})
                self.logger.log("assistant_response", {"content": assistant_content})
                self.logger.log_final(assistant_content)
                return assistant_content

            tool_result = process.open_url(url)
            self._log_tool_debug("open_url", {"url": url}, tool_result)
            self.logger.log(
                "tool_call",
                {
                    "tool_name": "open_url",
                    "tool_args": {"url": url},
                    "tool_result": tool_result,
                },
            )

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
            self.logger.log_llm_response(
                assistant_content,
                self._tool_calls_summary(getattr(final_response.choices[0].message, "tool_calls", []) or []),
            )
            self.messages.append({"role": "assistant", "content": assistant_content})
            self.logger.log("assistant_response", {"content": assistant_content})
            self.logger.log_final(assistant_content)
            return assistant_content

        self.messages.append({"role": "user", "content": user_input})
        tool_repeat_counts: dict[str, int] = {}
        total_tool_calls = 0

        for _ in range(10):
            response = self.client.chat(self.messages, TOOLS_SCHEMA)
            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)
            self.logger.log_llm_response(
                message.content or "",
                self._tool_calls_summary(tool_calls or []),
            )

            if tool_calls:
                for tool_call in tool_calls:
                    if total_tool_calls >= 8:
                        reason = "Превышен лимит инструментов (8). Пожалуйста, уточните задачу."
                        self.logger.warn("loop_guard: tool_call_limit")
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
                        self.logger.log_final(reason)
                        return reason

                    tool_name = tool_call.function.name
                    try:
                        args = json.loads(tool_call.function.arguments or "{}")
                    except json.JSONDecodeError:
                        args = {}
                    if not isinstance(args, dict):
                        args = {}
                    if tool_name in self._no_arg_tools:
                        args = {}
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
                        self.logger.warn("loop_guard: repeated_tool_call")
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
                        self.logger.log_final(assistant_content)
                        return assistant_content

                    level = risk_level(tool_name, args)
                    reason = risk_reason(tool_name, args, level)
                    approved = confirm_action(tool_name, args, level)
                    self.logger.log_policy(level.value, reason, approved)
                    self.logger.log(
                        "confirmation",
                        {
                            "tool_name": tool_name,
                            "tool_args": args,
                            "risk": level.value,
                            "risk_reason": reason,
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
                    self._log_tool_debug(tool_name, args, tool_result)
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
                    if parsed_result.get("ok") is False and parsed_result.get("error") == "use_open_url":
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
                        self.logger.log_llm_response(
                            assistant_content,
                            self._tool_calls_summary(getattr(final_response.choices[0].message, "tool_calls", []) or []),
                        )
                        self.messages.append({"role": "assistant", "content": assistant_content})
                        self.logger.log("assistant_response", {"content": assistant_content})
                        self.logger.log_final(assistant_content)
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
                        self.logger.log_llm_response(
                            assistant_content,
                            self._tool_calls_summary(getattr(final_response.choices[0].message, "tool_calls", []) or []),
                        )
                        self.messages.append({"role": "assistant", "content": assistant_content})
                        self.logger.log("assistant_response", {"content": assistant_content})
                        self.logger.log_final(assistant_content)
                        return assistant_content
                    total_tool_calls += 1
                continue

            assistant_content = message.content or ""
            self.messages.append({"role": "assistant", "content": assistant_content})
            self.logger.log("assistant_response", {"content": assistant_content})
            self.logger.log_final(assistant_content)
            return assistant_content

        fallback = "Reached tool execution limit without a final response."
        self.logger.log("assistant_response", {"content": fallback})
        self.logger.log_final(fallback)
        return fallback
