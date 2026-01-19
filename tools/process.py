from __future__ import annotations

import json
import os
import subprocess
from typing import Any

from tools.desktop import _take_screenshot


BLOCKLIST = ["del", "erase", "format", "diskpart", "rm -rf", "shutdown", "bcdedit"]


def _result(ok: bool, **kwargs: Any) -> str:
    payload = {"ok": ok, **kwargs}
    if ok:
        payload.setdefault("error", None)
    if not ok and "error" not in payload:
        payload["error"] = "unknown_error"
    return json.dumps(payload, ensure_ascii=False)


def _looks_like_path(app: str) -> bool:
    return any(sep in app for sep in ("/", "\\")) or app.lower().endswith(".exe")


def _parse_start_apps(output: str) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    lines = [line.rstrip() for line in output.splitlines() if line.strip()]
    if not lines:
        return items
    for line in lines[1:]:
        if line.startswith("-"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        app_id = parts[-1]
        name = " ".join(parts[:-1]).strip()
        if name and app_id:
            items.append({"name": name, "app_id": app_id})
    return items


def find_start_apps(query: str, limit: int = 10) -> str:
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-StartApps"],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return _result(False, error=completed.stderr.strip() or "powershell failed")
        items = _parse_start_apps(completed.stdout)
        if query:
            lowered = query.lower()
            items = [item for item in items if lowered in item["name"].lower()]
        return _result(True, items=items[:limit])
    except FileNotFoundError as exc:
        return _result(False, error=str(exc))
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc))


def open_start_app(app_id: str) -> str:
    try:
        subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{app_id}"])
        shot = _take_screenshot("after open_start_app")
        return _result(True, screenshot_path=shot["path"])
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc))


def open_app(app: str) -> str:
    try:
        if _looks_like_path(app):
            process = subprocess.Popen([app])
            shot = _take_screenshot(f"after open_app {app}")
            return _result(
                True,
                done=True,
                pid=getattr(process, "pid", None),
                screenshot_path=shot["path"],
                method="exe_path",
            )

        start_apps_raw = find_start_apps(app)
        start_apps = json.loads(start_apps_raw)
        if start_apps.get("ok") and start_apps.get("items"):
            items = start_apps["items"]
            lowered = app.lower()
            exact = next((item for item in items if item["name"].lower() == lowered), None)
            selection = exact or (items[0] if len(items) == 1 else None)
            if selection:
                opened = json.loads(open_start_app(selection["app_id"]))
                opened["done"] = True
                opened["method"] = "startapp"
                return json.dumps(opened, ensure_ascii=False)

        try:
            os.startfile(app)  # type: ignore[attr-defined]
            process = None
        except OSError:
            process = subprocess.Popen([app])
        shot = _take_screenshot(f"after open_app {app}")
        return _result(
            True,
            done=True,
            pid=getattr(process, "pid", None),
            screenshot_path=shot["path"],
            method="fallback",
        )
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc), method="fallback", done=True)


def open_url(url: str) -> str:
    try:
        os.startfile(url)  # type: ignore[attr-defined]
        method = "startfile"
    except OSError:
        subprocess.Popen(["cmd", "/c", "start", "", url])
        method = "cmd_start"
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc), url=url, done=True)

    try:
        shot = _take_screenshot("after open_url")
        return _result(
            True,
            url=url,
            done=True,
            method=method,
            screenshot_path=shot["path"],
        )
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc), url=url, done=True, method=method)


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
