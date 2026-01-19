from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any
import subprocess
import textwrap


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


def _run_command(command: list[str], exec_cmd: str, timeout_sec: int) -> str:
    start = time.perf_counter()
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
            verified=ok,
            verify_reason="returncode_zero" if ok else "nonzero_returncode",
            details={"exec": exec_cmd},
        )
    except subprocess.TimeoutExpired as exc:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return _result(
            False,
            returncode=None,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "timeout",
            duration_ms=duration_ms,
            verified=False,
            verify_reason="timeout",
            details={"exec": exec_cmd},
        )
    except Exception as exc:  # pragma: no cover - system dependent
        duration_ms = int((time.perf_counter() - start) * 1000)
        return _result(
            False,
            returncode=None,
            stdout="",
            stderr=str(exc),
            duration_ms=duration_ms,
            verified=False,
            verify_reason="exception",
            details={"exec": exec_cmd},
        )


def run_powershell(command: str, timeout_sec: int = 20) -> str:
    exec_cmd = f"powershell -NoProfile -ExecutionPolicy Bypass -Command \"{command}\""
    return _run_command(
        ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", command],
        exec_cmd,
        timeout_sec,
    )


def run_cmd(command: str, timeout_sec: int = 20) -> str:
    exec_cmd = f"cmd /c \"{command}\""
    return _run_command(["cmd", "/c", command], exec_cmd, timeout_sec)


def _wrap_python_code(code: str) -> str:
    indented = textwrap.indent(code.rstrip("\n"), "    ")
    return (
        "import traceback\n"
        "try:\n"
        f"{indented}\n"
        "except Exception:\n"
        "    traceback.print_exc()\n"
        "    raise\n"
    )


def run_python_script(code: str, timeout_sec: int = 20) -> str:
    scripts_dir = Path("scripts")
    scripts_dir.mkdir(parents=True, exist_ok=True)
    timestamp = int(time.time() * 1000)
    script_path = scripts_dir / f"tmp_{timestamp}.py"
    wrapped_code = _wrap_python_code(code)
    script_path.write_text(wrapped_code, encoding="utf-8")

    exec_cmd = f"python {script_path}"
    start = time.perf_counter()
    try:
        completed = subprocess.run(
            ["python", str(script_path)],
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
            error=None if ok else "python_error",
            duration_ms=duration_ms,
            verified=ok,
            verify_reason="returncode_zero" if ok else "nonzero_returncode",
            details={"exec": exec_cmd, "script_path": str(script_path)},
        )
    except subprocess.TimeoutExpired as exc:
        duration_ms = int((time.perf_counter() - start) * 1000)
        return _result(
            False,
            returncode=None,
            stdout=exc.stdout or "",
            stderr=exc.stderr or "timeout",
            error="timeout",
            duration_ms=duration_ms,
            verified=False,
            verify_reason="timeout",
            details={"exec": exec_cmd, "script_path": str(script_path)},
        )
    except Exception as exc:  # pragma: no cover - system dependent
        duration_ms = int((time.perf_counter() - start) * 1000)
        return _result(
            False,
            returncode=None,
            stdout="",
            stderr=str(exc),
            error=str(exc),
            duration_ms=duration_ms,
            verified=False,
            verify_reason="exception",
            details={"exec": exec_cmd, "script_path": str(script_path)},
        )
