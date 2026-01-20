from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Any
import subprocess


from .config import PROJECT_ROOT
from .state import add_recent_file, add_recent_url, set_active_file, set_active_url


SCRIPTS_DIR = PROJECT_ROOT / "scripts"
DESKTOP_PATH = Path.home() / "Desktop"


def _extract_url_from_powershell(code: str) -> str | None:
    match = re.search(r"Start-Process\s+['\"](https?://[^'\"]+)['\"]", code, re.IGNORECASE)
    if match:
        return match.group(1)
    return None


def _extract_printed_path(stdout: str) -> str | None:
    if not stdout:
        return None
    for line in stdout.splitlines()[::-1]:
        match = re.match(r"^(OK|SAVED):\s*(.+)$", line.strip(), re.IGNORECASE)
        if match:
            value = match.group(2).strip().strip('"').strip("'")
            return value or None
    return None


def _extract_desktop_path_from_python(code: str) -> str | None:
    pattern = re.compile(
        r'Path\.home\(\)\s*/\s*["\']Desktop["\']\s*/\s*["\']([^"\']+\.(?:docx|txt|xlsx|pdf|png|jpg))["\']',
        re.IGNORECASE,
    )
    match = pattern.search(code)
    if match:
        return str(DESKTOP_PATH / match.group(1))
    return None


def _track_result(language: str, code: str, result: dict[str, Any]) -> None:
    if not result.get("ok"):
        return
    stdout = result.get("stdout") or ""
    if language == "powershell":
        url = _extract_url_from_powershell(code)
        if url:
            add_recent_url(url)
            if "youtube.com" in url or "youtu.be" in url:
                set_active_url(url)
        return

    if language == "python":
        printed_path = _extract_printed_path(stdout)
        if printed_path:
            add_recent_file(printed_path)
            set_active_file(printed_path)
            return
        path = _extract_desktop_path_from_python(code)
        if path:
            add_recent_file(path)
            set_active_file(path)
            return


def _result(ok: bool, **kwargs: Any) -> dict[str, Any]:
    payload = {"ok": ok, **kwargs}
    if not ok and "error" not in payload:
        payload["error"] = "unknown_error"
    return payload


def _run_command(command: list[str], script_path: Path, timeout_sec: int, env: dict[str, str] | None = None) -> dict[str, Any]:
    start = time.perf_counter()
    exec_cmd = " ".join(command)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
            env=env,
        )
        duration_ms = int((time.perf_counter() - start) * 1000)
        ok = completed.returncode == 0
        stderr_text = completed.stderr or ""
        if not ok and "NoProcessFoundForGivenName" in stderr_text:
            ok = True
        return _result(
            ok,
            returncode=0 if ok else completed.returncode,
            stdout=completed.stdout,
            stderr=completed.stderr,
            duration_ms=duration_ms,
            script_path=str(script_path),
            exec_cmd=exec_cmd,
        )
    except subprocess.TimeoutExpired as exc:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return _result(
            False,
            returncode=None,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "timeout",
            duration_ms=duration_ms,
            script_path=str(script_path),
            exec_cmd=exec_cmd,
            error="timeout",
        )
    except Exception as exc:  # pragma: no cover - system dependent
        duration_ms = int((time.perf_counter() - start) * 1000)
        return _result(
            False,
            returncode=None,
            stdout="",
            stderr=str(exc),
            duration_ms=duration_ms,
            script_path=str(script_path),
            exec_cmd=exec_cmd,
            error=str(exc),
        )


def run_python(code: str, timeout_sec: int) -> dict[str, Any]:
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    script_path = SCRIPTS_DIR / f"tmp_{timestamp}.py"
    script_path.write_text(code, encoding="utf-8")
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    result = _run_command(["python", str(script_path)], script_path, timeout_sec, env=env)
    _track_result("python", code, result)
    return result


def run_powershell(code: str, timeout_sec: int) -> dict[str, Any]:
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    script_path = SCRIPTS_DIR / f"tmp_{timestamp}.ps1"
    script_path.write_text(code, encoding="utf-8")
    result = _run_command(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script_path)],
        script_path,
        timeout_sec,
    )
    _track_result("powershell", code, result)
    return result


def run_pip_install(package: str, timeout_sec: int) -> dict[str, Any]:
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    script_path = SCRIPTS_DIR / f"tmp_{timestamp}.cmd"
    command = ["python", "-m", "pip", "install", package]
    script_path.write_text(" ".join(command), encoding="utf-8")
    return _run_command(command, script_path, timeout_sec)
