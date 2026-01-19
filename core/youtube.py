from __future__ import annotations

import time
from urllib.parse import quote_plus

import pyautogui
import pygetwindow

YT_HOME = "https://www.youtube.com/"
YT_SEARCH_URL = "https://www.youtube.com/results?search_query={query}"
DEFAULT_VIDEO_URL = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"


def open_youtube_home() -> dict[str, str]:
    return {"language": "powershell", "script": f'Start-Process "{YT_HOME}"'}


def open_search(query: str) -> dict[str, str]:
    safe_query = quote_plus(query)
    return {
        "language": "powershell",
        "script": f'Start-Process "{YT_SEARCH_URL.format(query=safe_query)}"',
    }


def open_video_url(url: str) -> dict[str, str]:
    return {"language": "powershell", "script": f'Start-Process "{url}"'}


def open_first_video_from_search(query: str) -> dict[str, str]:
    return open_search(query)


def focus_browser_window() -> bool:
    keywords = ("youtube", "chrome", "firefox", "edge")
    for window in pygetwindow.getAllWindows():
        title = (window.title or "").lower()
        if any(keyword in title for keyword in keywords):
            try:
                if window.isMinimized:
                    window.restore()
                window.activate()
                time.sleep(0.2)
                return True
            except Exception:
                continue
    return False


def _press_repeated(key: str, count: int) -> None:
    for _ in range(max(1, count)):
        pyautogui.press(key)
        time.sleep(0.05)


def control(action: str, value: int | None = None) -> None:
    if not focus_browser_window():
        raise RuntimeError("Не найдено окно браузера/YouTube. Открой YouTube сначала.")

    pyautogui.FAILSAFE = True

    if action in {"play_pause", "pause"}:
        pyautogui.press("k")
    elif action == "seek_forward_10":
        _press_repeated("l", value or 1)
    elif action == "seek_back_10":
        _press_repeated("j", value or 1)
    elif action == "mute_toggle":
        pyautogui.press("m")
    elif action == "fullscreen_toggle":
        pyautogui.press("f")
    elif action == "next_video":
        pyautogui.hotkey("shift", "n")
    elif action == "prev_video":
        pyautogui.hotkey("shift", "p")
    elif action == "captions_toggle":
        pyautogui.press("c")
    elif action == "volume_up":
        _press_repeated("up", value or 1)
    elif action == "volume_down":
        _press_repeated("down", value or 1)
    elif action == "jump_percent":
        if value is None or not 0 <= value <= 9:
            raise ValueError("jump_percent requires value 0-9")
        pyautogui.press(str(value))
    elif action == "close_tab":
        pyautogui.hotkey("ctrl", "w")
    else:
        raise ValueError(f"Unknown action: {action}")
