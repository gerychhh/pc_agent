from __future__ import annotations

import importlib.util
import subprocess
import time
from typing import Any
from urllib.parse import urlparse

import pygetwindow


if importlib.util.find_spec("psutil"):
    import psutil  # type: ignore
else:  # pragma: no cover - optional dependency
    psutil = None


def _verify_result(verified: bool, reason: str, details: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "verified": verified,
        "reason": reason,
        "details": details or {},
    }


def wait_for_process(pid: int, timeout_sec: float = 5.0) -> bool:
    start = time.monotonic()
    while time.monotonic() - start <= timeout_sec:
        if psutil is not None:
            if psutil.pid_exists(pid):
                return True
        else:
            completed = subprocess.run(
                ["tasklist", "/FI", f"PID eq {pid}"],
                capture_output=True,
                text=True,
            )
            if completed.returncode == 0 and str(pid) in completed.stdout:
                return True
        time.sleep(0.2)
    return False


def find_window_title_contains(text: str) -> list[str]:
    query = (text or "").strip().lower()
    if not query:
        return []
    titles = pygetwindow.getAllTitles()
    return [title for title in titles if query in (title or "").lower()]


def wait_for_window_title(text: str, timeout_sec: float = 5.0) -> bool:
    start = time.monotonic()
    while time.monotonic() - start <= timeout_sec:
        if find_window_title_contains(text):
            return True
        time.sleep(0.2)
    return False


def verify_open_url(url: str) -> dict[str, Any]:
    parsed = urlparse(url)
    domain = parsed.netloc or parsed.path
    domain = domain.strip().strip("/")
    if not domain:
        return _verify_result(False, "missing_domain", {"url": url})
    verified = wait_for_window_title(domain, timeout_sec=5.0)
    if verified:
        titles = find_window_title_contains(domain)
        return _verify_result(True, "window_title_match", {"domain": domain, "titles": titles})
    return _verify_result(False, "window_not_found", {"domain": domain})
