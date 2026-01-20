from __future__ import annotations

from typing import Any


def build_command_index(commands: list[dict[str, Any]], limit: int = 25) -> str:
    lines: list[str] = []
    for command in commands[:limit]:
        cmd_id = command.get("id")
        desc = command.get("description_short") or ""
        intents = command.get("intents") or []
        intent_text = ", ".join(str(intent) for intent in intents[:3])
        lines.append(f"- {cmd_id}: {desc} | intents: {intent_text}")
    return "\n".join(lines)


def ctx_planner(
    state: dict[str, Any],
    command_index: str,
    user_text: str,
    last_run: list[dict[str, Any]] | None = None,
) -> str:
    last_payload = last_run or []
    return (
        "[SYSTEM] Ты — планировщик шагов для управления Windows-ПК. "
        "Верни один шаг в JSON без пояснений.\n"
        "[ENV] Windows 10/11, локальный запуск, нет system-role, ответ только user-message.\n"
        "[FORMAT] {\n"
        "  \"step_id\": \"...\",\n"
        "  \"use_command_id\": \"CMD_OPEN_NOTEPAD\" | null,\n"
        "  \"execute\": { \"lang\": \"powershell|python\", \"script\": \"...\" } | null,\n"
        "  \"verify\": { \"lang\": \"powershell|python\", \"script\": \"...\" } | null,\n"
        "  \"stop_if_failed\": true|false,\n"
        "  \"notes\": \"коротко\"\n"
        "}\n"
        "[RULES] Используй Command Library если возможно. После каждого шага обязательно verify.\n"
        f"[STATE] ACTIVE_FILE={state.get('active_file')} | ACTIVE_URL={state.get('active_url')} | "
        f"ACTIVE_APP={state.get('active_app')}\n"
        f"[COMMAND_INDEX]\n{command_index}\n"
        f"[TASK] {user_text}\n"
        f"[LAST_RUN] {last_payload}"
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
        "[ENV] Windows 10/11, локальный запуск.\n"
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
        f"[LAST_RUN] {last_exec}\n"
        f"[STEP] {step}"
    )


def ctx_reporter(state: dict[str, Any], results: list[dict[str, Any]], user_text: str) -> str:
    return (
        "[SYSTEM] Ты — репортёр выполнения команд на ПК. Отвечай кратко, без code blocks.\n"
        "[ENV] Windows 10/11.\n"
        f"[STATE] ACTIVE_FILE={state.get('active_file')} | ACTIVE_URL={state.get('active_url')} | "
        f"ACTIVE_APP={state.get('active_app')}\n"
        f"[TASK] {user_text}\n"
        f"[LAST_RUN] {results}\n"
        "[OUTPUT] 1-2 коротких предложения."
    )
