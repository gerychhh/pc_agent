from __future__ import annotations

import time
from pathlib import Path
from typing import Any
import subprocess


SCRIPTS_DIR = Path("scripts")


def _result(ok: bool, **kwargs: Any) -> dict[str, Any]:
    payload = {"ok": ok, **kwargs}
    if not ok and "error" not in payload:
        payload["error"] = "unknown_error"
    return payload


def _run_command(command: list[str], script_path: Path, timeout_sec: int) -> dict[str, Any]:
    start = time.perf_counter()
    exec_cmd = " ".join(command)
    try:
        completed = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        duration_ms = int((time.perf_counter() - start) * 1000)
        ok = completed.returncode == 0
        return _result(
            ok,
            returncode=completed.returncode,
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
    return _run_command(["python", str(script_path)], script_path, timeout_sec)


def run_powershell(code: str, timeout_sec: int) -> dict[str, Any]:
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    script_path = SCRIPTS_DIR / f"tmp_{timestamp}.ps1"
    script_path.write_text(code, encoding="utf-8")
    return _run_command(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-File", str(script_path)],
        script_path,
        timeout_sec,
    )


def run_pip_install(package: str, timeout_sec: int) -> dict[str, Any]:
    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    script_path = SCRIPTS_DIR / f"tmp_{timestamp}.cmd"
    command = ["python", "-m", "pip", "install", package]
    script_path.write_text(" ".join(command), encoding="utf-8")
    return _run_command(command, script_path, timeout_sec)
