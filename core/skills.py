from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus

import pyautogui
import pygetwindow


YT_HOME = "https://www.youtube.com/"
YT_SEARCH_URL = "https://www.youtube.com/results?search_query={query}"
DEFAULT_VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


@dataclass
class Action:
    language: str
    script: str
    updates: dict[str, Any] | None = None


def _focus_window(keywords: tuple[str, ...]) -> bool:
    for window in pygetwindow.getAllWindows():
        title = (window.title or "").lower()
        if any(keyword in title for keyword in keywords):
            if window.isMinimized:
                window.restore()
            window.activate()
            time.sleep(0.2)
            return True
    return False


def _youtube_window_open() -> bool:
    return _focus_window(("youtube", "chrome", "firefox", "edge"))


def _notepad_window_open() -> bool:
    return _focus_window(("notepad", "блокнот"))


def _powershell_start_process(target: str) -> str:
    return f'Start-Process "{target}"'


def match_skill(user_text: str, state: dict[str, Any]) -> Action | str | None:
    lowered = user_text.lower().strip()

    if "ютуб" in lowered or "youtube" in lowered or "на ютуб" in lowered:
        if lowered.startswith("открой ютуб") or lowered == "открой ютуб":
            return Action(
                language="powershell",
                script=_powershell_start_process(YT_HOME),
                updates={"active_url": YT_HOME},
            )

        search_match = re.search(r"найди на (?:ютубе|ютуб|youtube)\s+(.+)", user_text, re.IGNORECASE)
        if search_match:
            query = search_match.group(1).strip()
            url = YT_SEARCH_URL.format(query=quote_plus(query))
            return Action(
                language="powershell",
                script=_powershell_start_process(url),
                updates={"active_url": url},
            )

        if "включи видео" in lowered or "открой видео" in lowered:
            active_url = state.get("active_url") or ""
            if "results?search_query=" in active_url:
                url = DEFAULT_VIDEO_URL
            else:
                url = DEFAULT_VIDEO_URL
            return Action(
                language="powershell",
                script=_powershell_start_process(url),
                updates={"active_url": url},
            )

    if any(term in lowered for term in ("пауза", "стоп", "продолжи")) and (
        "ютуб" in lowered or "youtube" in lowered or state.get("active_url")
    ):
        if not _youtube_window_open():
            return "Открой YouTube сначала."
        script = _python_youtube_control("play_pause", 1)
        return Action(language="python", script=script)

    if (
        ("перемотай" in lowered and "вперед" in lowered)
        or ("впер" in lowered and ("сек" in lowered or "мин" in lowered))
    ) and ("ютуб" in lowered or state.get("active_url")):
        if not _youtube_window_open():
            return "Открой YouTube сначала."
        seconds = _extract_seconds(lowered) or 10
        steps = max(1, round(seconds / 10))
        return Action(language="python", script=_python_youtube_control("seek_forward_10", steps))

    if "назад" in lowered and ("ютуб" in lowered or state.get("active_url")):
        if not _youtube_window_open():
            return "Открой YouTube сначала."
        seconds = _extract_seconds(lowered) or 10
        steps = max(1, round(seconds / 10))
        return Action(language="python", script=_python_youtube_control("seek_back_10", steps))

    if "выключи звук" in lowered and ("ютуб" in lowered or state.get("active_url")):
        if not _youtube_window_open():
            return "Открой YouTube сначала."
        return Action(language="python", script=_python_youtube_control("mute_toggle", 1))

    if "громче" in lowered and ("ютуб" in lowered or state.get("active_url")):
        if not _youtube_window_open():
            return "Открой YouTube сначала."
        return Action(language="python", script=_python_youtube_control("volume_up", 3))

    if "тише" in lowered and ("ютуб" in lowered or state.get("active_url")):
        if not _youtube_window_open():
            return "Открой YouTube сначала."
        return Action(language="python", script=_python_youtube_control("volume_down", 3))

    if "на весь экран" in lowered and ("ютуб" in lowered or state.get("active_url")):
        if not _youtube_window_open():
            return "Открой YouTube сначала."
        return Action(language="python", script=_python_youtube_control("fullscreen_toggle", 1))

    if "открой блокнот" in lowered:
        return Action(
            language="powershell",
            script="Start-Process notepad.exe",
            updates={"active_app": "notepad"},
        )

    if "закрой блокнот" in lowered:
        return Action(
            language="powershell",
            script="Stop-Process -Name notepad -ErrorAction SilentlyContinue",
            updates={"active_app": "notepad"},
        )

    write_match = re.search(r"напиши в (?:этом )?блокноте\s+(.+)", user_text, re.IGNORECASE)
    if write_match:
        if not _notepad_window_open():
            return "Открой блокнот сначала."
        text = write_match.group(1).strip()
        return Action(language="python", script=_python_type_in_notepad(text))

    url_match = re.search(r"открой\s+([a-z0-9.-]+\\.[a-z]{2,})(/\\S*)?$", lowered)
    if url_match:
        url = url_match.group(1)
        if not url.startswith(("http://", "https://")):
            url = f"https://{url}"
        return Action(language="powershell", script=_powershell_start_process(url), updates={"active_url": url})

    app_aliases = {"калькулятор": "calc.exe", "calc": "calc.exe"}
    for alias, exe in app_aliases.items():
        if f"открой {alias}" in lowered:
            return Action(
                language="powershell",
                script=f"Start-Process {exe}",
                updates={"active_app": exe},
            )

    return None


def _extract_seconds(text: str) -> int | None:
    match = re.search(r"(\\d+)\\s*(сек|секунд|секунды|с)", text)
    if match:
        return int(match.group(1))
    match = re.search(r"(\\d+)\\s*(мин|минута|минуты|минут)", text)
    if match:
        return int(match.group(1)) * 60
    if "мин" in text and not re.search(r"\\d", text):
        return 60
    return None


def _python_youtube_control(action: str, steps: int) -> str:
    return (
        "import time\n"
        "import pyautogui\n"
        "import pygetwindow\n"
        "\n"
        "def focus():\n"
        "    for window in pygetwindow.getAllWindows():\n"
        "        title = (window.title or \"\").lower()\n"
        "        if any(key in title for key in (\"youtube\", \"chrome\", \"firefox\", \"edge\")):\n"
        "            if window.isMinimized:\n"
        "                window.restore()\n"
        "            window.activate()\n"
        "            time.sleep(0.2)\n"
        "            return True\n"
        "    return False\n"
        "\n"
        "if not focus():\n"
        "    raise RuntimeError(\"Не найдено окно браузера/YouTube. Открой YouTube сначала.\")\n"
        "pyautogui.FAILSAFE = True\n"
        f"action = \"{action}\"\n"
        f"steps = {steps}\n"
        "def press_many(key, count):\n"
        "    for _ in range(max(1, count)):\n"
        "        pyautogui.press(key)\n"
        "        time.sleep(0.05)\n"
        "if action == \"play_pause\":\n"
        "    pyautogui.press(\"k\")\n"
        "elif action == \"seek_forward_10\":\n"
        "    press_many(\"l\", steps)\n"
        "elif action == \"seek_back_10\":\n"
        "    press_many(\"j\", steps)\n"
        "elif action == \"mute_toggle\":\n"
        "    pyautogui.press(\"m\")\n"
        "elif action == \"volume_up\":\n"
        "    press_many(\"up\", steps)\n"
        "elif action == \"volume_down\":\n"
        "    press_many(\"down\", steps)\n"
        "elif action == \"fullscreen_toggle\":\n"
        "    pyautogui.press(\"f\")\n"
    )


def _python_type_in_notepad(text: str) -> str:
    safe = text.replace("\\", "\\\\").replace("\"", "\\\"")
    return (
        "import time\n"
        "import pyautogui\n"
        "import pygetwindow\n"
        "\n"
        "def focus():\n"
        "    for window in pygetwindow.getAllWindows():\n"
        "        title = (window.title or \"\").lower()\n"
        "        if \"notepad\" in title or \"блокнот\" in title:\n"
        "            if window.isMinimized:\n"
        "                window.restore()\n"
        "            window.activate()\n"
        "            time.sleep(0.2)\n"
        "            return True\n"
        "    return False\n"
        "\n"
        "if not focus():\n"
        "    raise RuntimeError(\"Не найдено окно блокнота. Открой блокнот сначала.\")\n"
        "pyautogui.typewrite(\"" + safe + "\", interval=0.01)\n"
    )
