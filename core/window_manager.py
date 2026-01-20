from __future__ import annotations

from typing import Any

import psutil
import pyautogui
import win32gui
import win32process

from .debug import debug_event


def get_active_window_info() -> dict[str, Any]:
    hwnd = win32gui.GetForegroundWindow()
    title = win32gui.GetWindowText(hwnd) if hwnd else ""
    pid = 0
    process_name = ""
    if hwnd:
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        if pid:
            process_name = _safe_process_name(pid)
    return {
        "title": title,
        "hwnd": int(hwnd or 0),
        "pid": int(pid or 0),
        "process": process_name,
    }


def list_windows() -> list[dict[str, Any]]:
    windows: list[dict[str, Any]] = []

    def _enum_handler(hwnd: int, result: list[dict[str, Any]]) -> None:
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if not title:
            return
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        result.append(
            {
                "title": title,
                "hwnd": int(hwnd),
                "pid": int(pid or 0),
                "process": _safe_process_name(pid),
            }
        )

    win32gui.EnumWindows(_enum_handler, windows)
    return windows


def focus_window_by_title_contains(substr: str) -> bool:
    target = substr.strip().lower()
    if not target:
        return False
    log_active_window("WIN_BEFORE")
    for window in list_windows():
        title = window.get("title", "")
        if target in title.lower():
            return _focus_hwnd(window.get("hwnd"), f"title_contains:{substr}")
    return False


def focus_window_by_process(process_name: str) -> bool:
    target = process_name.strip().lower()
    if not target:
        return False
    log_active_window("WIN_BEFORE")
    for window in list_windows():
        proc = (window.get("process") or "").lower()
        if proc == target:
            return _focus_hwnd(window.get("hwnd"), f"process:{process_name}")
    return False


def click_center_active_window() -> None:
    info = log_active_window("WIN_BEFORE")
    hwnd = info.get("hwnd")
    if not hwnd:
        raise RuntimeError("No active window")
    left, top, right, bottom = win32gui.GetWindowRect(hwnd)
    center_x = int((left + right) / 2)
    center_y = int((top + bottom) / 2)
    pyautogui.click(center_x, center_y)
    debug_event("WIN_CLICK", f"x={center_x} y={center_y}")
    log_active_window("WIN_AFTER")


def ensure_focus(target: dict[str, str]) -> bool:
    title = target.get("title_contains")
    if title:
        if focus_window_by_title_contains(title):
            return True
    process = target.get("process")
    if process:
        if focus_window_by_process(process):
            return True
    debug_event("WIN_FOCUS", f"success=false target={target}")
    return False


def press_hotkey(*keys: str) -> None:
    log_active_window("WIN_BEFORE")
    pyautogui.hotkey(*keys)
    debug_event("WIN_KEY", f"key={'+'.join(keys)}")
    log_active_window("WIN_AFTER")


def press_key(key: str) -> None:
    log_active_window("WIN_BEFORE")
    pyautogui.press(key)
    debug_event("WIN_KEY", f"key={key}")
    log_active_window("WIN_AFTER")


def type_text(text: str, interval: float = 0.01) -> None:
    log_active_window("WIN_BEFORE")
    pyautogui.typewrite(text, interval=interval)
    debug_event("WIN_KEY", "key=type_text")
    log_active_window("WIN_AFTER")


def log_active_window(tag: str) -> dict[str, Any]:
    info = get_active_window_info()
    debug_event(
        tag,
        f"title=\"{info.get('title','')}\" process=\"{info.get('process','')}\" pid={info.get('pid')} hwnd={info.get('hwnd')}",
    )
    return info


def log_verify(ok: bool, details: str = "") -> None:
    status = "true" if ok else "false"
    message = f"ok={status}"
    if details:
        message = f"{message} details={details}"
    debug_event("WIN_VERIFY", message)


def _safe_process_name(pid: int | None) -> str:
    if not pid:
        return ""
    try:
        return psutil.Process(pid).name() or ""
    except Exception:
        return ""


def _focus_hwnd(hwnd: int | None, target_desc: str) -> bool:
    if not hwnd:
        debug_event("WIN_FOCUS", f"success=false target={target_desc}")
        return False
    try:
        win32gui.ShowWindow(hwnd, 9)
        win32gui.SetForegroundWindow(hwnd)
        debug_event("WIN_FOCUS", f"success=true target={target_desc}")
        log_active_window("WIN_AFTER")
        return True
    except Exception as exc:
        debug_event("WIN_FOCUS", f"success=false target={target_desc} error={exc}")
        return False
