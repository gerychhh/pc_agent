from __future__ import annotations

from typing import Any


def build_command_index(commands: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for command in commands:
        cmd_id = command.get("id")
        desc = command.get("description_short") or ""
        intents = command.get("intents") or []
        intent_text = ", ".join(str(intent) for intent in intents[:4])
        lines.append(f"- {cmd_id}: {desc} | intents: {intent_text}")
    return "\n".join(lines)


def ctx_fast(state: dict[str, Any], command_index: str, user_text: str) -> str:
    return (
        "[SYSTEM] Ты — локальный ассистент управления Windows-ПК. "
        "Отвечай кратко и без code blocks.\n"
        f"[STATE] ACTIVE_FILE={state.get('active_file')} | ACTIVE_URL={state.get('active_url')} | "
        f"ACTIVE_APP={state.get('active_app')}\n"
        f"[COMMAND_INDEX]\n{command_index}\n"
        f"[TASK] {user_text}\n"
        "[OUTPUT] Напиши 1-2 предложения результата."
    )


def ctx_planner(state: dict[str, Any], command_index: str, user_text: str) -> str:
    return (
        "[SYSTEM] Ты — планировщик шагов для управления Windows-ПК. "
        "Верни ОДИН следующий шаг в JSON без пояснений. "
        "Если задача уже выполнена, верни шаг с execute: null и verify: null, notes: 'done'.\n"
        "[FORMAT] {\n"
        "  \"step_id\": \"...\",\n"
        "  \"use_command_id\": \"CMD_OPEN_NOTEPAD\" | null,\n"
        "  \"execute\": { \"lang\": \"powershell|python\", \"script\": \"...\" } | null,\n"
        "  \"verify\": { \"lang\": \"powershell|python\", \"script\": \"...\" } | null,\n"
        "  \"stop_if_failed\": true|false,\n"
        "  \"notes\": \"коротко\"\n"
        "}\n"
        "[RULES] Используй Command Library, если подходит. "
        "Если нужно набрать текст в блокноте, используй python + pyautogui и проверку активного окна "
        "(pygetwindow.getActiveWindow().title содержит Notepad/Блокнот). "
        "Для закрытия YouTube вкладок: найди окно браузера с YouTube, активируй и делай Ctrl+W пока заголовок не YouTube.\n"
        f"[STATE] ACTIVE_FILE={state.get('active_file')} | ACTIVE_URL={state.get('active_url')} | "
        f"ACTIVE_APP={state.get('active_app')}\n"
        f"[COMMAND_INDEX]\n{command_index}\n"
        f"[TASK] {user_text}"
    )


def ctx_smart_fix(
    state: dict[str, Any],
    command_index: str,
    step: dict[str, Any],
    last_exec: dict[str, Any],
    user_text: str,
) -> str:
    return (
        "[SYSTEM] Исправь шаг, который упал. Верни новый шаг JSON без пояснений.\n"
        "[FORMAT] {\n"
        "  \"step_id\": \"...\",\n"
        "  \"use_command_id\": \"CMD_OPEN_NOTEPAD\" | null,\n"
        "  \"execute\": { \"lang\": \"powershell|python\", \"script\": \"...\" },\n"
        "  \"verify\": { \"lang\": \"powershell|python\", \"script\": \"...\" },\n"
        "  \"stop_if_failed\": true|false,\n"
        "  \"notes\": \"коротко\"\n"
        "}\n"
        f"[STATE] ACTIVE_FILE={state.get('active_file')} | ACTIVE_URL={state.get('active_url')} | "
        f"ACTIVE_APP={state.get('active_app')}\n"
        f"[COMMAND_INDEX]\n{command_index}\n"
        f"[TASK] {user_text}\n"
        f"[STEP] {step}\n"
        f"[LAST_RESULT] OK={last_exec.get('ok')} stdout={last_exec.get('stdout')} stderr={last_exec.get('stderr')}"
    )


def ctx_reporter(state: dict[str, Any], results: list[dict[str, Any]], user_text: str) -> str:
    return (
        "[SYSTEM] Ты — репортёр выполнения команд на ПК. Отвечай кратко, без code blocks.\n"
        f"[STATE] ACTIVE_FILE={state.get('active_file')} | ACTIVE_URL={state.get('active_url')} | "
        f"ACTIVE_APP={state.get('active_app')}\n"
        f"[TASK] {user_text}\n"
        f"[RESULTS] {results}\n"
        "[OUTPUT] 1-2 коротких предложения."
    )
