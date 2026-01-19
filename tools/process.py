from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from core.config import SCREENSHOT_DIR
from tools.desktop import _take_screenshot


BLOCKLIST = ["del", "erase", "format", "diskpart", "rm -rf", "shutdown", "bcdedit"]


def _result(ok: bool, **kwargs: Any) -> str:
    payload = {"ok": ok, **kwargs}
    if not ok and "error" not in payload:
        payload["error"] = "unknown_error"
    return json.dumps(payload, ensure_ascii=False)


def _looks_like_path(app: str) -> bool:
    return any(sep in app for sep in ("/", "\\")) or app.lower().endswith(".exe")


def open_app(app: str) -> str:
    try:
        if _looks_like_path(app):
            process = subprocess.Popen([app])
        else:
            try:
                os.startfile(app)  # type: ignore[attr-defined]
                process = None
            except OSError:
                process = subprocess.Popen([app])
        shot = _take_screenshot(f"after open_app {app}")
        return _result(True, pid=getattr(process, "pid", None), screenshot_path=shot["path"])
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc))


def run_cmd(cmd: str, timeout_sec: int = 15) -> str:
    lowered = cmd.lower()
    if any(blocked in lowered for blocked in BLOCKLIST):
        return _result(False, error="blocked by policy")

    try:
        completed = subprocess.run(
            cmd,
            shell=True,
            timeout=timeout_sec,
            capture_output=True,
            text=True,
        )
        shot = _take_screenshot(f"after run_cmd {cmd}")
        return _result(
            completed.returncode == 0,
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
            screenshot_path=shot["path"],
        )
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc))
