from __future__ import annotations

import time

import pyautogui
import pyperclip

from .debug import debug_event
from .window_manager import click_center_active_window


def windows_search_open(query: str) -> None:
    for attempt in range(1, 3):
        debug_event("APP_OPEN", f"strategy=windows_search attempt={attempt}")
        try:
            pyautogui.hotkey("win", "s")
            time.sleep(0.40)
            try:
                click_center_active_window()
            except Exception:
                pass
            pyperclip.copy(query)
            pyautogui.hotkey("ctrl", "v")
            time.sleep(0.15)
            pyautogui.press("enter")
            time.sleep(0.60)
            print(f"[APP_OPEN] attempt={attempt} ok=true")
            break
        except Exception as exc:
            print(f"[APP_OPEN] attempt={attempt} ok=false error={exc}")
