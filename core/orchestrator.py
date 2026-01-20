from __future__ import annotations

import json
import re
from typing import Any

from .command_engine import CommandResult, load_commands, match_command, run_command
from .config import FAST_MODEL, TIMEOUT_SEC
from .context_builder import ctx_action
from .debug import debug_event, truncate_text
from .llm_client import LLMClient
from .executor import run_powershell, run_python
from .state import load_state, update_active_window_state
from .validator import validate_powershell, validate_python


BLOCKED_MESSAGE = "Команда заблокирована по безопасности (опасная операция)."


def sanitize_assistant_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("[TOOL_RESULT]", "").replace("[/TOOL_RESULT]", "")
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


class Orchestrator:
    def __init__(self) -> None:
        self.client = LLMClient()
        self.commands = load_commands()

    def reset(self) -> None:
        return None

    def run(self, user_text: str, stateless: bool = False, force_llm: bool = False) -> str:
        state = load_state()
        debug_event("USER_IN", user_text)
        if not force_llm:
            match, _ = match_command(user_text, self.commands)
            if match:
                action_desc = _format_command_action(match.command.get("id"), match.params)
                result = run_command(match.command, match.params)
                summarized = _summarize_command_result(result)
                return _format_action_output(action_desc, _format_simple_response(summarized))

        llm_action = self._run_llm_script(user_text, state)
        if llm_action is None:
            return "Не удалось подобрать команду."
        llm_result, action_desc = llm_action
        if llm_result.get("blocked"):
            print("неудачная операция")
            return BLOCKED_MESSAGE
        if not llm_result.get("ok"):
            print("неудачная операция")
        return _format_action_output(action_desc, _format_simple_response(llm_result))

    def _run_llm_script(self, user_text: str, state: dict[str, Any]) -> tuple[dict[str, Any], str] | None:
        payload = ctx_action(state, user_text)
        debug_event("LLM_REQ", f"action model={FAST_MODEL} payload={truncate_text(payload, 400)}")
        try:
            response = self.client.chat(
                [{"role": "user", "content": payload}],
                tools=[],
                model_name=FAST_MODEL,
                tool_choice="none",
            )
            content = response.choices[0].message.content or ""
        except Exception as exc:
            debug_event("LLM_ERR", str(exc))
            return None

        action = _parse_action_json(content)
        if not action:
            return None

        execute = action.get("execute") or {}
        if not execute:
            return None
        lang = execute.get("lang")
        script = execute.get("script")
        if not lang or not script:
            return None
        errors = validate_python(script) if lang == "python" else validate_powershell(script)
        if errors:
            return {"id": "LLM_SCRIPT", "ok": False, "stderr": "\n".join(errors), "blocked": True}, _format_script_action(lang, script)
        result = run_python(script, TIMEOUT_SEC) if lang == "python" else run_powershell(script, TIMEOUT_SEC)
        update_active_window_state()
        return {"id": "LLM_SCRIPT", "ok": bool(result.get("ok")), "stdout": result.get("stdout"), "stderr": result.get("stderr")}, _format_script_action(lang, script)


def _format_command_action(command_id: str | None, params: dict[str, str]) -> str:
    if not command_id:
        return "COMMAND: unknown"
    if not params:
        return f"COMMAND: {command_id}"
    params_view = ", ".join(f"{key}={value}" for key, value in params.items())
    return f"COMMAND: {command_id} ({params_view})"


def _format_script_action(lang: str | None, script: str | None) -> str:
    language = lang or "script"
    snippet = (script or "").strip().replace("\n", " ")
    if len(snippet) > 120:
        snippet = snippet[:120] + "..."
    return f"SCRIPT[{language}]: {snippet}"


def _format_action_output(action: str, response: str) -> str:
    return f"{action}\n{response}".strip()


def _summarize_command_result(result: CommandResult) -> dict[str, Any]:
    return {
        "id": result.action.name,
        "ok": result.ok,
        "stdout": result.execute_result.stdout,
        "stderr": result.execute_result.stderr,
    }


def _format_simple_response(result: dict[str, Any]) -> str:
    if result.get("ok"):
        return "Готово."
    stderr = result.get("stderr")
    if stderr:
        return f"Не удалось выполнить запрос. {stderr}".strip()
    return "Не удалось выполнить запрос."


def _parse_action_json(content: str) -> dict[str, Any] | None:
    cleaned = re.sub(r"```.*?```", "", content, flags=re.DOTALL).strip()
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return None
