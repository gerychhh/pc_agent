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


def ctx_action(state: dict[str, Any], command_index: str, user_text: str) -> str:
    return (
        "[SYSTEM] Ты — исполнитель для управления Windows-ПК. "
        "Верни один JSON без пояснений.\n"
        "[ENV] Windows 10/11, локальный запуск.\n"
        "[FORMAT] {\n"
        "  \"use_command_id\": \"CMD_OPEN_NOTEPAD\" | null,\n"
        "  \"params\": {\"text\": \"...\"} | null,\n"
        "  \"execute\": { \"lang\": \"powershell|python\", \"script\": \"...\" } | null,\n"
        "  \"verify\": { \"lang\": \"powershell|python\", \"script\": \"...\" } | null\n"
        "}\n"
        "[RULES]\n"
        "- Если есть подходящая команда — ОБЯЗАТЕЛЬНО верни use_command_id и params.\n"
        "- Если нет команды — верни execute (и verify при необходимости).\n"
        "- НЕ используй Start-Process для открытия приложений. Для запуска приложений используй CMD_OPEN_APP_SEARCH.\n"
        "- Для поиска в интернете используй CMD_BROWSER_SEARCH или CMD_OPEN_URL.\n"
        "- Не используй опасные действия: удаление, shutdown, реестр, системные каталоги.\n"
        "- Пример: \"открой блокнот\" -> use_command_id=CMD_OPEN_APP_SEARCH, params={\"app_name\":\"блокнот\"}.\n"
        f"[STATE] ACTIVE_FILE={state.get('active_file')} | ACTIVE_URL={state.get('active_url')} | "
        f"ACTIVE_APP={state.get('active_app')}\n"
        f"[COMMAND_INDEX]\n{command_index}\n"
        f"[TASK] {user_text}\n"
        "[OUTPUT] Верни только JSON."
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
