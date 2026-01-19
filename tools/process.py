from __future__ import annotations

import json
import os
import subprocess
import difflib
from datetime import datetime, timezone
from typing import Any

from core.app_cache import (
    get_cached_launch,
    invalidate_cached_launch,
    update_cached_launch,
)
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


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _is_valid_cached_launch(record: dict[str, Any]) -> bool:
    launch_type = record.get("launch_type")
    target = record.get("target")
    if not isinstance(target, str) or not target:
        return False
    if launch_type == "exe":
        return target.lower().endswith(".exe") and os.path.exists(target)
    if launch_type == "shortcut":
        return target.lower().endswith(".lnk") and os.path.exists(target)
    if launch_type == "startapp":
        return True
    return False


def _normalize_shortcut_items(parsed: Any) -> list[dict[str, str]]:
    items: list[dict[str, str]] = []
    if isinstance(parsed, dict):
        parsed_items = [parsed]
    elif isinstance(parsed, list):
        parsed_items = parsed
    else:
        parsed_items = []
    for it in parsed_items:
        name = str(it.get("name", "")).strip()
        path = str(it.get("path", "")).strip()
        if name and path:
            items.append({"name": name, "path": path})
    return items


def _find_shortcuts_in_paths(query: str, paths: list[str], limit: int = 10) -> list[dict[str, str]]:
    if not query or not paths:
        return []
    existing_paths = [p for p in paths if p and os.path.isdir(p)]
    if not existing_paths:
        return []
    q = query.replace('"', "")
    ps_paths = ",".join(f'"{p}"' for p in existing_paths)
    ps = (
        "$q=\"" + q + "\"; "
        f"$paths=@({ps_paths}); "
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
        return []
    out = completed.stdout.strip()
    if not out:
        return []
    try:
        parsed = json.loads(out)
    except json.JSONDecodeError:
        return []
    return _normalize_shortcut_items(parsed)[:limit]


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
        paths = [
            os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs"),
            os.path.join(os.environ.get("ProgramData", ""), "Microsoft", "Windows", "Start Menu", "Programs"),
        ]
        items = _find_shortcuts_in_paths(query, paths, limit=limit)
        return _result(True, items=items)
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc))


def open_start_app(app_id: str) -> str:
    try:
        subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{app_id}"])
        shot = _take_screenshot("after open_start_app")
        return _result(
            True,
            screenshot_path=shot["path"],
            method="startapp",
            launch_type="startapp",
            target=app_id,
        )
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc))


def open_app(app: str) -> str:
    try:
        cached = get_cached_launch(app)
        if cached and not _is_valid_cached_launch(cached):
            invalidate_cached_launch(app)
            cached = None
        if cached:
            launch_type = cached.get("launch_type")
            target = cached.get("target")
            try:
                if launch_type == "exe":
                    process = subprocess.Popen([target])
                elif launch_type == "shortcut":
                    os.startfile(target)  # type: ignore[attr-defined]
                    process = None
                elif launch_type == "startapp":
                    opened = json.loads(open_start_app(target))
                    if not opened.get("ok"):
                        invalidate_cached_launch(app)
                        cached = None
                    else:
                        refreshed = dict(cached)
                        refreshed["last_verified_utc"] = _utc_now()
                        update_cached_launch(app, refreshed)
                        shot = opened.get("screenshot_path")
                        return _result(
                            True,
                            app=app,
                            method="cache",
                            launch_type="startapp",
                            target=target,
                            pid=None,
                            screenshot_path=shot,
                        )
                else:
                    cached = None
                    process = None

                if cached and launch_type in ("exe", "shortcut"):
                    refreshed = dict(cached)
                    refreshed["last_verified_utc"] = _utc_now()
                    update_cached_launch(app, refreshed)
                    shot = _take_screenshot(f"after open_app cache {app}")
                    return _result(
                        True,
                        app=app,
                        method="cache",
                        launch_type=launch_type,
                        target=target,
                        pid=getattr(process, "pid", None),
                        screenshot_path=shot["path"],
                    )
            except Exception:
                invalidate_cached_launch(app)

        if _looks_like_path(app) and app.lower().endswith(".exe") and os.path.exists(app):
            process = subprocess.Popen([app])
            shot = _take_screenshot(f"after open_app {app}")
            record = {
                "display_name": os.path.basename(app),
                "launch_type": "exe",
                "target": app,
                "last_verified_utc": _utc_now(),
                "confidence": 1.0,
            }
            update_cached_launch(app, record)
            return _result(
                True,
                app=app,
                method="discovered",
                launch_type="exe",
                target=app,
                pid=getattr(process, "pid", None),
                screenshot_path=shot["path"],
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
                if opened.get("ok"):
                    record = {
                        "display_name": selection["name"],
                        "launch_type": "startapp",
                        "target": selection["app_id"],
                        "last_verified_utc": _utc_now(),
                        "confidence": 1.0 if exact else 0.8,
                    }
                    update_cached_launch(app, record)
                    return _result(
                        True,
                        app=app,
                        method="discovered",
                        launch_type="startapp",
                        target=selection["app_id"],
                        pid=None,
                        screenshot_path=opened.get("screenshot_path"),
                    )

        pinned_paths = [
            os.path.join(
                os.environ.get("APPDATA", ""),
                "Microsoft",
                "Internet Explorer",
                "Quick Launch",
                "User Pinned",
                "StartMenu",
            )
        ]
        pinned_shortcuts = _find_shortcuts_in_paths(app, pinned_paths, limit=3)
        if pinned_shortcuts:
            path = pinned_shortcuts[0]["path"]
            os.startfile(path)  # type: ignore[attr-defined]
            shot = _take_screenshot(f"after open_pinned_start {path}")
            record = {
                "display_name": pinned_shortcuts[0]["name"],
                "launch_type": "shortcut",
                "target": path,
                "last_verified_utc": _utc_now(),
                "confidence": 0.8,
            }
            update_cached_launch(app, record)
            return _result(
                True,
                app=app,
                method="discovered",
                launch_type="shortcut",
                target=path,
                pid=None,
                screenshot_path=shot["path"],
            )

        desktop_paths = [
            os.path.join(os.environ.get("USERPROFILE", ""), "Desktop"),
            os.path.join(os.environ.get("PUBLIC", ""), "Desktop"),
        ]
        desktop_shortcuts = _find_shortcuts_in_paths(app, desktop_paths, limit=3)
        if desktop_shortcuts:
            path = desktop_shortcuts[0]["path"]
            os.startfile(path)  # type: ignore[attr-defined]
            shot = _take_screenshot(f"after open_desktop_shortcut {path}")
            record = {
                "display_name": desktop_shortcuts[0]["name"],
                "launch_type": "shortcut",
                "target": path,
                "last_verified_utc": _utc_now(),
                "confidence": 0.75,
            }
            update_cached_launch(app, record)
            return _result(
                True,
                app=app,
                method="discovered",
                launch_type="shortcut",
                target=path,
                pid=None,
                screenshot_path=shot["path"],
            )

        shortcuts_raw = find_start_menu_shortcuts(app, limit=3)
        shortcuts = json.loads(shortcuts_raw)
        if shortcuts.get("ok") and shortcuts.get("items"):
            path = shortcuts["items"][0]["path"]
            os.startfile(path)  # type: ignore[attr-defined]
            shot = _take_screenshot(f"after open_shortcut {path}")
            record = {
                "display_name": shortcuts["items"][0]["name"],
                "launch_type": "shortcut",
                "target": path,
                "last_verified_utc": _utc_now(),
                "confidence": 0.7,
            }
            update_cached_launch(app, record)
            return _result(
                True,
                app=app,
                method="discovered",
                launch_type="shortcut",
                target=path,
                pid=None,
                screenshot_path=shot["path"],
            )

        return _result(
            False,
            app=app,
            error="app_not_found",
            method="fallback",
            launch_type=None,
            target=None,
            user_hint=(
                f"Не могу найти приложение '{app}'. Укажи путь к .exe или точное название как в меню Пуск."
            ),
        )
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(
            False,
            app=app,
            error=str(exc),
            method="fallback",
            launch_type=None,
            target=None,
        )


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
