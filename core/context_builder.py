from __future__ import annotations

from typing import Any


def ctx_action(state: dict[str, Any], user_text: str) -> str:
    return (
        "[SYSTEM] Ты — помощник для управления Windows-ПК. "
        "Верни один JSON без пояснений.\n"
        "[ENV] Windows 10/11, локальный запуск.\n"
        "[FORMAT] {\n"
        "  \"execute\": { \"lang\": \"powershell|python\", \"script\": \"...\" }\n"
        "  # или составной вариант:\n"
        "  \"execute\": { \"steps\": [ {\"lang\": \"powershell|python\", \"script\": \"...\"} ] }\n"
        "}\n"
        "[RULES]\n"
        "- Всегда возвращай execute (single или steps).\n"
        "- Скрипт должен выполнять задачу пользователя.\n"
        "- Для открытия сайтов используй PowerShell Start-Process с URL.\n"
        "- Для открытия приложений используй поиск через меню Пуск (start).\n"
        "- Для YouTube:\n"
        "  * Поиск: открой https://www.youtube.com/results?search_query=... через Start-Process.\n"
        "  * Управление видео: используй Python + pyautogui, сфокусируй окно браузера по заголовку "
        "('youtube', 'chrome', 'firefox', 'edge'), затем нажимай клавиши:\n"
        "    - k: play/pause, j/l: назад/вперед 10 сек, m: mute, f: fullscreen, up/down: громкость.\n"
        "  * Для 'включи видео' просто сфокусируй окно и нажми k.\n"
        "  * Пример:\n"
        "    import time, pyautogui, pygetwindow\n"
        "    for w in pygetwindow.getAllWindows():\n"
        "        if any(k in (w.title or '').lower() for k in ('youtube','chrome','firefox','edge')):\n"
        "            w.activate(); time.sleep(0.2); break\n"
        "    pyautogui.press('k')\n"
        "- Шаблоны системных действий:\n"
        "  * Громкость Windows (пример: установить 30%):\n"
        "    import subprocess\n"
        "    level = 30\n"
        "    subprocess.run(['powershell','-NoProfile','-Command',"
        "     f'(New-Object -ComObject WScript.Shell).SendKeys([char]173); "
        "     1..50 | % { (New-Object -ComObject WScript.Shell).SendKeys([char]174) }; "
        "     1..{int(level/2)} | % { (New-Object -ComObject WScript.Shell).SendKeys([char]175) }'],"
        "     check=False)\n"
        "    # 173 mute, 174 volume down, 175 volume up\n"
        "  * Скриншот активного окна:\n"
        "    import pyautogui, time\n"
        "    time.sleep(0.2)\n"
        "    pyautogui.hotkey('alt','prtsc')\n"
        "    # далее сохранить буфер через стандартные средства\n"
        "  * Закрытие активного окна:\n"
        "    import pyautogui\n"
        "    pyautogui.hotkey('alt','f4')\n"
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
