from __future__ import annotations

import json
import os
import subprocess
import difflib
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

        # Intelligent matching:
        # 1) substring match
        # 2) fuzzy match via stdlib difflib
        if query:
            q = query.strip().lower()
            substring = [item for item in items if q in item["name"].lower()]
            if substring:
                items = substring
            else:
                names = [item["name"] for item in items]
                close = difflib.get_close_matches(query, names, n=limit, cutoff=0.55)
                if close:
                    close_set = {c.lower() for c in close}
                    items = [item for item in items if item["name"].lower() in close_set]

        return _result(True, items=items[:limit])
    except FileNotFoundError as exc:
        return _result(False, error=str(exc))
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc))


def find_start_menu_shortcuts(query: str, limit: int = 10) -> str:
    """Find Start Menu shortcuts (.lnk) matching query (helps with apps not in Get-StartApps)."""
    try:
        q = query.replace('"', "")
        ps = (
            "$q=\"" + q + "\"; "
            "$paths=@($env:APPDATA+'\\Microsoft\\Windows\\Start Menu\\Programs',"
            "$env:ProgramData+'\\Microsoft\\Windows\\Start Menu\\Programs'); "
            "$items=Get-ChildItem $paths -Recurse -Filter *.lnk -ErrorAction SilentlyContinue "
            "| Where-Object { $_.BaseName -like ('*'+$q+'*') } "
            "| Select-Object -First " + str(max(limit, 1)) + " @{'n'='name';'e'={$_.BaseName}},@{'n'='path';'e'={$_.FullName}}; "
            "$items | ConvertTo-Json -Compress"
        )
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return _result(False, error=completed.stderr.strip() or "powershell failed")

        out = completed.stdout.strip()
        if not out:
            return _result(True, items=[])

        parsed = json.loads(out)
        # PowerShell returns object or array depending on count
        if isinstance(parsed, dict):
            items = [parsed]
        elif isinstance(parsed, list):
            items = parsed
        else:
            items = []

        normalized = []
        for it in items:
            name = str(it.get("name", "")).strip()
            path = str(it.get("path", "")).strip()
            if name and path:
                normalized.append({"name": name, "path": path})

        return _result(True, items=normalized[:limit])
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc))


def open_start_app(app_id: str) -> str:
    try:
        subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{app_id}"])
        shot = _take_screenshot("after open_start_app")
        return _result(True, screenshot_path=shot["path"], method="startapp")
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc))


def open_app(app: str) -> str:
    try:
        if _looks_like_path(app):
            process = subprocess.Popen([app])
            shot = _take_screenshot(f"after open_app {app}")
            return _result(
                True,
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
                return json.dumps(opened, ensure_ascii=False)

        # Fallback: search Start Menu shortcuts (.lnk)
        shortcuts_raw = find_start_menu_shortcuts(app, limit=3)
        shortcuts = json.loads(shortcuts_raw)
        if shortcuts.get("ok") and shortcuts.get("items"):
            path = shortcuts["items"][0]["path"]
            try:
                os.startfile(path)  # type: ignore[attr-defined]
                shot = _take_screenshot(f"after open_shortcut {path}")
                return _result(True, pid=None, screenshot_path=shot["path"], method="startmenu_shortcut")
            except Exception as exc:
                # continue to final fallback
                pass

        try:
            os.startfile(app)  # type: ignore[attr-defined]
            process = None
        except OSError:
            process = subprocess.Popen([app])
        shot = _take_screenshot(f"after open_app {app}")
        return _result(
            True,
            pid=getattr(process, "pid", None),
            screenshot_path=shot["path"],
            method="fallback",
        )
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc), method="fallback")


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
