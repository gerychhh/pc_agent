from __future__ import annotations

import time

import pyautogui
import pyperclip

from .debug import debug_event


def windows_search_open(query: str) -> None:
    for attempt in range(1, 3):
        debug_event("APP_OPEN", f"strategy=windows_search attempt={attempt}")
        try:
            pyautogui.press("win")
            time.sleep(0.45)
            pyperclip.copy(query)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.20)
            pyautogui.press("enter")
            time.sleep(0.60)
            print(f"[APP_OPEN] attempt={attempt} ok=true")
            break
        except Exception as exc:
            print(f"[APP_OPEN] attempt={attempt} ok=false error={exc}")
