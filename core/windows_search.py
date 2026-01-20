from __future__ import annotations

import time

import pyautogui
import pyperclip

from .debug import debug_event
from .window_manager import get_active_window_info


_SEARCH_PROCESSES = {
    "searchapp.exe",
    "searchhost.exe",
    "startmenuexperiencehost.exe",
}


def _is_search_focused() -> bool:
    info = get_active_window_info()
    process = (info.get("process") or "").lower()
    title = (info.get("title") or "").lower()
    if process in _SEARCH_PROCESSES:
        return True
    return "search" in title or "поиск" in title or "пуск" in title


def windows_search_open(query: str) -> None:
    for attempt in range(1, 3):
        debug_event("APP_OPEN", f"strategy=windows_search attempt={attempt}")
        try:
            pyautogui.hotkey("win", "s")
            time.sleep(0.45)
            if not _is_search_focused():
                pyautogui.press("win")
                time.sleep(0.35)
            if _is_search_focused():
                pyautogui.hotkey("ctrl", "a")
                time.sleep(0.05)
            pyperclip.copy(query)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.20)
            pyautogui.press("enter")
            time.sleep(0.60)
            print(f"[APP_OPEN] attempt={attempt} ok=true")
            break
        except Exception as exc:
            print(f"[APP_OPEN] attempt={attempt} ok=false error={exc}")
