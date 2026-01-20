from __future__ import annotations

import re
from typing import Any

from .command_engine import CommandResult, load_commands, match_command, run_command
from .config import FAST_MODEL, SMART_MODEL
from .context_builder import build_command_index, ctx_reporter
from .debug import debug_event, truncate_text
from .llm_client import LLMClient
from .planner_loop import PlannerLoop
from .state import load_state
from .window_manager import ensure_focus, get_active_window_info


BLOCKED_MESSAGE = "Команда заблокирована по безопасности (опасная операция)."

PAUSE_WORDS = ("пауза", "поставь на паузу", "останови видео", "play", "pause", "продолжи")
SEEK_BACK_WORDS = ("назад", "перемотай назад", "отмотай", "перемотай на", "перемотай обратно")
SEEK_FORWARD_WORDS = ("вперёд", "вперед", "перемотай вперед", "перемотай вперёд", "промотай")
BROWSER_PROCESSES = ("chrome.exe", "msedge.exe", "firefox.exe")


def sanitize_assistant_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("[TOOL_RESULT]", "").replace("[/TOOL_RESULT]", "")
    cleaned = re.sub(r"```.*?```", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def _contains_multiple_actions(text: str) -> bool:
    lowered = text.lower()
    return any(token in lowered for token in (" и ", " затем ", " потом ", ","))


def split_into_subtasks(text: str) -> list[str]:
    lowered = text.lower()
    for token in (" затем ", " потом ", " и "):
        if token in lowered:
            parts = [part.strip() for part in re.split(rf"\\s*{re.escape(token.strip())}\\s*", text) if part.strip()]
            return parts
    return [text.strip()] if text.strip() else []


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
        self.commands = load_commands()
        self.command_index = build_command_index(self.commands)
        self.planner = PlannerLoop()

    def reset(self) -> None:
        return None

    def run(self, user_text: str, stateless: bool = False) -> str:
        state = load_state()
        debug_event("USER_IN", user_text)

        media_response = self._handle_media_control(user_text, state)
        if media_response is not None:
            return media_response

        best_match, matches = match_command(user_text, self.commands)
        has_multiple = _contains_multiple_actions(user_text)
        if best_match and best_match.score >= 2 and not has_multiple:
            debug_event("ROUTER", "simple -> command")
            result = run_command(best_match.command, best_match.params)
            if _is_blocked_result(result):
                return BLOCKED_MESSAGE
            return self._report(user_text, state, [_summarize_command_result(result)])

        if best_match and best_match.score == 3 and "*" in best_match.intent:
            debug_event("ROUTER", "wildcard -> command")
            result = run_command(best_match.command, best_match.params)
            if _is_blocked_result(result):
                return BLOCKED_MESSAGE
            return self._report(user_text, state, [_summarize_command_result(result)])

        if has_multiple:
            subtasks = split_into_subtasks(user_text)
            if subtasks:
                results: list[dict[str, Any]] = []
                for subtask in subtasks:
                    sub_match, _ = match_command(subtask, self.commands)
                    if not sub_match or sub_match.score < 2:
                        results = []
                        break
                    debug_event("ROUTER", f"subtask -> command: {sub_match.command.get('id')}")
                    sub_result = run_command(sub_match.command, sub_match.params)
                    if _is_blocked_result(sub_result):
                        return BLOCKED_MESSAGE
                    results.append(_summarize_command_result(sub_result))
                if results:
                    return self._report(user_text, state, results)

        if not _is_action_like(user_text):
            # Не действие. Раньше мы просто эхо-повторяли текст пользователя — это выглядело как баг.
            debug_event("ROUTER", "not an action -> assistant")
            lowered = user_text.strip().lower()
            if any(token in lowered for token in ("не сделал", "не получилось", "не работает", "ошибка")):
                return "Понял. Давай попробуем ещё раз — скажи точную команду, что сделать на компьютере."
            if lowered in {"что", "чего", "а", "?", "ээ", "эм", "мм"}:
                return "Скажи команду: открой/закрой приложение, напиши текст, найди видео на YouTube и т.д."
            return "Ок. Скажи команду, что сделать на компьютере."

        debug_event("ROUTER", "complex -> planner")
        plan_results = self.planner.run(user_text, state, self.command_index)
        if not plan_results:
            return "Не удалось построить план."
        if any(item.blocked for item in plan_results):
            return BLOCKED_MESSAGE

        summary = [_summarize_step_execution(step) for step in plan_results]
        return self._report(user_text, state, summary)

    def _handle_media_control(self, user_text: str, state: dict[str, Any]) -> str | None:
        lowered = user_text.lower()
        if not _is_media_control(lowered):
            return None
        if not _ensure_browser_focus():
            return "Не нашёл окно браузера, открой YouTube и повтори."
        if any(token in lowered for token in PAUSE_WORDS):
            return self._run_command_by_id("CMD_YT_TOGGLE_PLAY", {}, user_text, state)
        if any(token in lowered for token in SEEK_BACK_WORDS):
            seconds = _parse_seek_seconds(lowered) or 10
            return self._run_command_by_id("CMD_YT_BACK_SECONDS", {"seconds": str(seconds)}, user_text, state)
        if any(token in lowered for token in SEEK_FORWARD_WORDS):
            seconds = _parse_seek_seconds(lowered) or 10
            return self._run_command_by_id("CMD_YT_FORWARD_SECONDS", {"seconds": str(seconds)}, user_text, state)
        return None

    def _run_command_by_id(self, command_id: str, params: dict[str, str], user_text: str, state: dict[str, Any]) -> str:
        command = next((cmd for cmd in self.commands if cmd.get("id") == command_id), None)
        if not command:
            return "Не нашёл команду для управления медиа."
        result = run_command(command, params)
        if _is_blocked_result(result):
            return BLOCKED_MESSAGE
        return self._report(user_text, state, [_summarize_command_result(result)])

    def _report(self, user_text: str, state: dict[str, Any], results: list[dict[str, Any]]) -> str:
        payload = ctx_reporter(state, results, user_text)
        model = SMART_MODEL if any(not r.get("ok") for r in results) else FAST_MODEL
        debug_event("LLM_REQ", f"report model={model} payload={truncate_text(payload, 400)}")
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


def _is_media_control(lowered: str) -> bool:
    return any(token in lowered for token in PAUSE_WORDS + SEEK_BACK_WORDS + SEEK_FORWARD_WORDS)


def _ensure_browser_focus() -> bool:
    active = get_active_window_info()
    active_title = (active.get("title") or "").lower()
    active_process = (active.get("process") or "").lower()
    if "youtube" in active_title or active_process in BROWSER_PROCESSES:
        return True
    if ensure_focus({"title_contains": "YouTube"}):
        return True
    for process in BROWSER_PROCESSES:
        if ensure_focus({"process": process}):
            return True
    return False


def _parse_seek_seconds(text: str) -> int | None:
    match = re.search(r"(\\d+)\\s*(секунд|сек|с)", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(\\d+)\\s*(минут|минуты|минута|мин)", text)
    if match:
        return int(match.group(1)) * 60
    if "минут" in text or "минута" in text:
        return 60
    return None


def _summarize_command_result(result: CommandResult) -> dict[str, Any]:
    return {
        "id": result.action.name,
        "ok": result.ok,
        "stdout": result.execute_result.stdout,
        "stderr": result.execute_result.stderr,
    }


def _summarize_step_execution(step: Any) -> dict[str, Any]:
    return {
        "step_id": step.step.get("step_id"),
        "ok": step.ok,
        "stdout": (step.execute_result or {}).get("stdout"),
        "stderr": (step.execute_result or {}).get("stderr"),
    }


def _is_blocked_result(result: CommandResult) -> bool:
    if result.execute_result.error == "blocked":
        return True
    stderr = result.execute_result.stderr or ""
    return "запрещ" in stderr.lower()
