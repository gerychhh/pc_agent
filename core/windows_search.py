from __future__ import annotations

import time
from ctypes import Structure, WinDLL, sizeof
from ctypes.wintypes import DWORD, LONG, WORD

import pyautogui
import pyperclip

from .debug import debug_event
from .window_manager import get_active_window_info


_SEARCH_PROCESSES = {
    "searchapp.exe",
    "searchhost.exe",
    "startmenuexperiencehost.exe",
}


class _KEYBDINPUT(Structure):
    _fields_ = [
        ("wVk", WORD),
        ("wScan", WORD),
        ("dwFlags", DWORD),
        ("time", DWORD),
        ("dwExtraInfo", LONG),
    ]


class _INPUT(Structure):
    _fields_ = [("type", DWORD), ("ki", _KEYBDINPUT)]


_INPUT_KEYBOARD = 1
_KEYEVENTF_UNICODE = 0x0004
_KEYEVENTF_KEYUP = 0x0002
_USER32 = WinDLL("user32", use_last_error=True)


def _is_search_focused() -> bool:
    info = get_active_window_info()
    process = (info.get("process") or "").lower()
    title = (info.get("title") or "").lower()
    if process in _SEARCH_PROCESSES:
        return True
    return "search" in title or "поиск" in title or "пуск" in title


def _send_unicode_text(text: str) -> None:
    for char in text:
        key_down = _INPUT(type=_INPUT_KEYBOARD, ki=_KEYBDINPUT(wVk=0, wScan=ord(char), dwFlags=_KEYEVENTF_UNICODE))
        key_up = _INPUT(
            type=_INPUT_KEYBOARD,
            ki=_KEYBDINPUT(wVk=0, wScan=ord(char), dwFlags=_KEYEVENTF_UNICODE | _KEYEVENTF_KEYUP),
        )
        inputs = (_INPUT * 2)(key_down, key_up)
        _USER32.SendInput(2, inputs, sizeof(_INPUT))


def _should_type_unicode(text: str) -> bool:
    return any(ord(char) > 127 for char in text)


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
                pyautogui.press("backspace")
            pyperclip.copy(query)
            if _should_type_unicode(query):
                _send_unicode_text(query)
            else:
                pyautogui.hotkey("ctrl", "v")
            time.sleep(0.20)
            pyautogui.press("enter")
            time.sleep(0.60)
            print(f"[APP_OPEN] attempt={attempt} ok=true")
            break
        except Exception as exc:
            print(f"[APP_OPEN] attempt={attempt} ok=false error={exc}")
