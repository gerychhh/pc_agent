from __future__ import annotations

from typing import Any


def ctx_action(state: dict[str, Any], user_text: str) -> str:
    return (
        "[SYSTEM] Ты — помощник для управления Windows-ПК. "
        "Верни один JSON без пояснений.\n"
        "[ENV] Windows 10/11, локальный запуск.\n"
        "[FORMAT] {\n"
        "  \"execute\": { \"lang\": \"powershell|python\", \"script\": \"...\" }\n"
        "}\n"
        "[RULES]\n"
        "- Всегда возвращай execute.\n"
        "- Скрипт должен выполнять задачу пользователя.\n"
        "- Для открытия сайтов используй PowerShell Start-Process с URL.\n"
        "- Для открытия приложений используй поиск через меню Пуск (start).\n"
        "- Запрещено удалять файлы и папки.\n"
        f"[STATE] ACTIVE_FILE={state.get('active_file')} | ACTIVE_URL={state.get('active_url')} | "
        f"ACTIVE_APP={state.get('active_app')}\n"
        f"[TASK] {user_text}\n"
        "[OUTPUT] Верни только JSON."
    )


def ctx_reporter(state: dict[str, Any], results: list[dict[str, Any]], user_text: str) -> str:
    return (
        "[SYSTEM] Ты — репортёр выполнения команд на ПК. Отвечай кратко по-русски, без code blocks.\n"
        "[ENV] Windows 10/11.\n"
        f"[STATE] ACTIVE_FILE={state.get('active_file')} | ACTIVE_URL={state.get('active_url')} | "
        f"ACTIVE_APP={state.get('active_app')}\n"
        f"[TASK] {user_text}\n"
        f"[LAST_RUN] {results}\n"
        "[OUTPUT] 1-2 коротких предложения."
    )
