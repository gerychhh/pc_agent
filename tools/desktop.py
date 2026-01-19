from __future__ import annotations

import importlib.util
import json
from datetime import datetime
from typing import Any

import pyautogui

from core.config import SCREENSHOT_DIR


if importlib.util.find_spec("cv2"):
    import cv2  # type: ignore
else:  # pragma: no cover - optional dependency
    cv2 = None


pyautogui.FAILSAFE = True


def _result(ok: bool, **kwargs: Any) -> str:
    payload = {"ok": ok, **kwargs}
    if ok:
        payload.setdefault("error", None)
    if not ok and "error" not in payload:
        payload["error"] = "unknown_error"
    if "verified" not in payload:
        payload["verified"] = ok
    payload.setdefault("verify_reason", None)
    payload.setdefault("details", {})
    return json.dumps(payload, ensure_ascii=False)


def _take_screenshot(note: str) -> dict[str, Any]:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = SCREENSHOT_DIR / f"shot_{timestamp}.png"
    image = pyautogui.screenshot()
    image.save(path)
    return {
        "path": str(path),
        "width": image.width,
        "height": image.height,
        "note": note,
    }


def screenshot(note: str = "") -> str:
    try:
        payload = _take_screenshot(note)
        return _result(True, **payload)
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc))


def move_mouse(x: int, y: int) -> str:
    try:
        pyautogui.moveTo(x, y)
        return _result(True)
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc))


def click(x: int, y: int, button: str = "left", clicks: int = 1, interval: float = 0.1) -> str:
    try:
        pyautogui.click(x=x, y=y, button=button, clicks=clicks, interval=interval)
        shot = _take_screenshot(f"after click {x},{y}")
        return _result(
            True,
            screenshot_path=shot["path"],
            verified=False,
            verify_reason="no_click_verification_without_ocr",
        )
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc))


def type_text(text: str, interval: float = 0.02) -> str:
    try:
        pyautogui.write(text, interval=interval)
        shot = _take_screenshot("after type_text")
        return _result(
            True,
            screenshot_path=shot["path"],
            verified=False,
            verify_reason="no_text_verification_without_ocr",
        )
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc))


def press_key(key: str) -> str:
    try:
        pyautogui.press(key)
        shot = _take_screenshot(f"after press_key {key}")
        return _result(
            True,
            screenshot_path=shot["path"],
            verified=False,
            verify_reason="no_key_verification_without_ocr",
        )
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc))


def hotkey(keys: list[str]) -> str:
    try:
        pyautogui.hotkey(*keys)
        shot = _take_screenshot(f"after hotkey {'+'.join(keys)}")
        return _result(
            True,
            screenshot_path=shot["path"],
            verified=False,
            verify_reason="no_hotkey_verification_without_ocr",
        )
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc))


def locate_on_screen(image_path: str, confidence: float = 0.8) -> str:
    try:
        if confidence is not None and cv2 is None:
            return _result(False, error="opencv not installed", verified=False, verify_reason="missing_opencv")
        location = pyautogui.locateCenterOnScreen(image_path, confidence=confidence)
        if location is None:
            return _result(False, error="not found", verified=False, verify_reason="not_found")
        return _result(True, x=location.x, y=location.y)
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc))
