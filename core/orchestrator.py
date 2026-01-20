from __future__ import annotations

import re
from typing import Any

from .command_engine import CommandResult, load_commands, match_command, run_command
from .config import FAST_MODEL, SMART_MODEL
from .context_builder import build_command_index, ctx_fast, ctx_reporter
from .llm_client import LLMClient
from .logger import SessionLogger
from .planner_loop import PlannerLoop
from .state import load_state


BLOCKED_MESSAGE = "Команда заблокирована по безопасности (опасная операция)."


def sanitize_assistant_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("[TOOL_RESULT]", "").replace("[/TOOL_RESULT]", "")
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _contains_multiple_actions(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in (" и ", " затем ", " потом ", ","))


def _is_action_like(user_input: str) -> bool:
    verbs = (
        "открой",
        "закрой",
        "найди",
        "включи",
        "запусти",
        "создай",
        "измени",
        "сделай",
        "удали",
        "скачай",
        "перемотай",
        "напиши",
        "впиши",
    )
    lowered = user_input.lower()
    return any(verb in lowered for verb in verbs)


class Orchestrator:
    def __init__(self) -> None:
        self.client = LLMClient()
        self.logger = SessionLogger()
        self.commands = load_commands()
        self.command_index = build_command_index(self.commands)
        self.planner = PlannerLoop()

    def run(self, user_text: str, stateless: bool = False) -> str:
        state = load_state()
        self.logger.log_user_input(user_text, 1)
        match = match_command(user_text, self.commands)
        if match and not match.missing and not _contains_multiple_actions(user_text):
            result = run_command(match)
            if _is_blocked_result(result):
                return BLOCKED_MESSAGE
            return self._report(user_text, state, [_summarize_command_result(result)])

        if not match and not _is_action_like(user_text):
            return self._fast_reply(user_text, state)

        plan_results = self.planner.run(user_text, state, self.command_index)
        if not plan_results:
            return "Не удалось построить план."
        if any(item.blocked for item in plan_results):
            return BLOCKED_MESSAGE

        summary = [_summarize_step_execution(step) for step in plan_results]
        return self._report(user_text, state, summary)

    def _report(self, user_text: str, state: dict[str, Any], results: list[dict[str, Any]]) -> str:
        payload = ctx_reporter(state, results, user_text)
        model = SMART_MODEL if any(not r.get("ok") for r in results) else FAST_MODEL
        try:
            response = self.client.chat(
                [{"role": "user", "content": payload}],
                tools=[],
                model_name=model,
                tool_choice="none",
            )
            content = response.choices[0].message.content or ""
            return sanitize_assistant_text(content)
        except Exception:
            if all(r.get("ok") for r in results):
                return "Готово."
            return "Не удалось выполнить запрос."

    def _fast_reply(self, user_text: str, state: dict[str, Any]) -> str:
        payload = ctx_fast(state, self.command_index, user_text)
        try:
            response = self.client.chat(
                [{"role": "user", "content": payload}],
                tools=[],
                model_name=FAST_MODEL,
                tool_choice="none",
            )
            content = response.choices[0].message.content or ""
            return sanitize_assistant_text(content)
        except Exception:
            return "Уточни, что нужно сделать."


def _summarize_command_result(result: CommandResult) -> dict[str, Any]:
    return {
        "id": result.action.name,
        "ok": result.ok,
        "stdout": result.execute_result.get("stdout"),
        "stderr": result.execute_result.get("stderr"),
    }


def _summarize_step_execution(step: Any) -> dict[str, Any]:
    return {
        "step_id": step.step.get("step_id"),
        "ok": step.ok,
        "stdout": (step.execute_result or {}).get("stdout"),
        "stderr": (step.execute_result or {}).get("stderr"),
    }


def _is_blocked_result(result: CommandResult) -> bool:
    stderr = result.execute_result.get("stderr") or ""
    if result.execute_result.get("error") == "blocked":
        return True
    return "запрещ" in stderr.lower()
