from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from typing import Any
from urllib.parse import urlparse

from core.app_cache import get_cached_launch, invalidate_cached_launch, update_cached_launch
from tools import verify
from tools.desktop import _take_screenshot


BLOCKLIST = ["del", "erase", "format", "diskpart", "rm -rf", "shutdown", "bcdedit"]


def _result(ok: bool, **kwargs: Any) -> str:
    payload = {"ok": ok, **kwargs}
    if ok:
        payload.setdefault("error", None)
    if not ok and "error" not in payload:
        payload["error"] = "unknown_error"
    if "verified" not in payload:
        payload["verified"] = ok
    payload.setdefault("verify_reason", None)
    payload.setdefault("verify_details", {})
    payload.setdefault("details", {})
    return json.dumps(payload, ensure_ascii=False)


def _looks_like_path(app: str) -> bool:
    return any(sep in app for sep in ("/", "\\")) or app.lower().endswith(".exe")


def _looks_like_url(text: str) -> bool:
    if not text:
        return False
    if re.match(r"^https?://", text, re.IGNORECASE):
        return True
    parsed = urlparse(text)
    if parsed.scheme and parsed.netloc:
        return True
    return bool(re.search(r"\bwww\.[^\s]+", text, re.IGNORECASE))


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


def _verify_launch(pid: int | None, window_title: str | None) -> dict[str, Any]:
    if pid is not None:
        verified = verify.wait_for_process(pid)
        return {
            "verified": verified,
            "verify_reason": "process_detected" if verified else "process_not_found",
            "verify_details": {"pid": pid},
        }
    if window_title:
        verified = verify.wait_for_window_title(window_title)
        return {
            "verified": verified,
            "verify_reason": "window_title_match" if verified else "window_not_found",
            "verify_details": {"window_title": window_title},
        }
    return {
        "verified": False,
        "verify_reason": "missing_verification_target",
        "verify_details": {},
    }


def find_start_apps(query: str, limit: int = 10) -> str:
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", "Get-StartApps"],
            capture_output=True,
            text=True,
        )
        if completed.returncode != 0:
            return _result(False, error=completed.stderr.strip() or "powershell failed", verified=False)
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

        return _result(True, items=items[:limit], verified=True)
    except FileNotFoundError as exc:
        return _result(False, error=str(exc), verified=False)
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc), verified=False)


def find_start_menu_shortcuts(query: str, limit: int = 10) -> str:
    """Find Start Menu shortcuts (.lnk) matching query (helps with apps not in Get-StartApps)."""
    try:
        paths = [
            os.path.join(os.environ.get("APPDATA", ""), "Microsoft", "Windows", "Start Menu", "Programs"),
            os.path.join(os.environ.get("ProgramData", ""), "Microsoft", "Windows", "Start Menu", "Programs"),
        ]
        items = _find_shortcuts_in_paths(query, paths, limit=limit)
        return _result(True, items=items, verified=True)
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc), verified=False)


def open_start_app(app_id: str, display_name: str | None = None) -> str:
    try:
        subprocess.Popen(["explorer.exe", f"shell:AppsFolder\\{app_id}"])
        shot = _take_screenshot("after open_start_app")
        verification = _verify_launch(None, display_name)
        return _result(
            True,
            screenshot_path=shot["path"],
            method="startapp",
            launch_type="startapp",
            target=app_id,
            verified=verification["verified"],
            verify_reason=verification["verify_reason"],
            verify_details=verification["verify_details"],
            details={"display_name": display_name},
        )
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc), verified=False)


def open_app(app: str) -> str:
    try:
        if _looks_like_url(app):
            return _result(
                False,
                app=app,
                error="use_open_url_tool",
                verified=False,
                verify_reason="url_detected",
                details={"hint": "Use open_url for URLs"},
            )

        cached = get_cached_launch(app)
        if cached and not _is_valid_cached_launch(cached):
            invalidate_cached_launch(app)
            cached = None
        if cached:
            launch_type = cached.get("launch_type")
            target = cached.get("target")
            display_name = cached.get("display_name") or app
            try:
                if launch_type == "exe":
                    process = subprocess.Popen([target])
                    pid = getattr(process, "pid", None)
                    shot = _take_screenshot(f"after open_app cache {app}")
                    verification = _verify_launch(pid, display_name)
                elif launch_type == "shortcut":
                    os.startfile(target)  # type: ignore[attr-defined]
                    pid = None
                    shot = _take_screenshot(f"after open_app cache {app}")
                    verification = _verify_launch(None, display_name)
                elif launch_type == "startapp":
                    opened = json.loads(open_start_app(target, display_name))
                    if not opened.get("ok"):
                        invalidate_cached_launch(app)
                        cached = None
                    else:
                        verification = {
                            "verified": opened.get("verified"),
                            "verify_reason": opened.get("verify_reason"),
                            "verify_details": opened.get("verify_details"),
                        }
                        shot = opened.get("screenshot_path")
                        pid = None
                else:
                    cached = None
                    verification = {"verified": False, "verify_reason": "invalid_cache", "verify_details": {}}
                    pid = None
                    shot = None

                if cached and verification["verified"]:
                    refreshed = dict(cached)
                    refreshed["last_verified_utc"] = _utc_now()
                    update_cached_launch(app, refreshed)
                    return _result(
                        True,
                        app=app,
                        method="cache",
                        launch_type=launch_type,
                        target=target,
                        pid=pid,
                        screenshot_path=shot,
                        verified=True,
                        verify_reason=verification["verify_reason"],
                        verify_details=verification["verify_details"],
                    )
                if cached and not verification["verified"]:
                    invalidate_cached_launch(app)
            except Exception:
                invalidate_cached_launch(app)

        if _looks_like_path(app) and app.lower().endswith(".exe") and os.path.exists(app):
            process = subprocess.Popen([app])
            pid = getattr(process, "pid", None)
            shot = _take_screenshot(f"after open_app {app}")
            verification = _verify_launch(pid, os.path.basename(app))
            if verification["verified"]:
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
                pid=pid,
                screenshot_path=shot["path"],
                verified=verification["verified"],
                verify_reason=verification["verify_reason"],
                verify_details=verification["verify_details"],
            )

        start_apps_raw = find_start_apps(app)
        start_apps = json.loads(start_apps_raw)
        if start_apps.get("ok") and start_apps.get("items"):
            items = start_apps["items"]
            lowered = app.lower()
            exact = next((item for item in items if item["name"].lower() == lowered), None)
            selection = exact or (items[0] if len(items) == 1 else None)
            if selection:
                opened = json.loads(open_start_app(selection["app_id"], selection["name"]))
                if opened.get("ok"):
                    verification = {
                        "verified": opened.get("verified"),
                        "verify_reason": opened.get("verify_reason"),
                        "verify_details": opened.get("verify_details"),
                    }
                    if verification["verified"]:
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
                        verified=verification["verified"],
                        verify_reason=verification["verify_reason"],
                        verify_details=verification["verify_details"],
                    )

        shortcuts_raw = find_start_menu_shortcuts(app, limit=3)
        shortcuts = json.loads(shortcuts_raw)
        if shortcuts.get("ok") and shortcuts.get("items"):
            path = shortcuts["items"][0]["path"]
            display_name = shortcuts["items"][0]["name"]
            os.startfile(path)  # type: ignore[attr-defined]
            shot = _take_screenshot(f"after open_shortcut {path}")
            verification = _verify_launch(None, display_name)
            if verification["verified"]:
                record = {
                    "display_name": display_name,
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
                verified=verification["verified"],
                verify_reason=verification["verify_reason"],
                verify_details=verification["verify_details"],
            )

        return _result(
            False,
            app=app,
            error="app_not_found",
            method="fallback",
            verified=False,
            verify_reason="not_found",
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
            verified=False,
            verify_reason="exception",
        )


def open_url(url: str) -> str:
    try:
        os.startfile(url)  # type: ignore[attr-defined]
        method = "startfile"
    except OSError:
        subprocess.Popen(["cmd", "/c", "start", "", url])
        method = "cmd_start"
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc), url=url, done=True, verified=False, verify_reason="exception")

    try:
        shot = _take_screenshot("after open_url")
        verification = verify.verify_open_url(url)
        return _result(
            True,
            url=url,
            done=True,
            method=method,
            screenshot_path=shot["path"],
            verified=verification["verified"],
            verify_reason=verification["reason"],
            verify_details=verification["details"],
        )
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(
            False,
            error=str(exc),
            url=url,
            done=True,
            method=method,
            verified=False,
            verify_reason="exception",
        )


def run_cmd(cmd: str, timeout_sec: int = 15) -> str:
    lowered = cmd.lower()
    if any(blocked in lowered for blocked in BLOCKLIST):
        return _result(False, error="blocked by policy", verified=False, verify_reason="blocked")

    try:
        completed = subprocess.run(
            cmd,
            shell=True,
            timeout=timeout_sec,
            capture_output=True,
            text=True,
        )
        shot = _take_screenshot(f"after run_cmd {cmd}")
        ok = completed.returncode == 0
        return _result(
            ok,
            stdout=completed.stdout,
            stderr=completed.stderr,
            returncode=completed.returncode,
            screenshot_path=shot["path"],
            verified=ok,
            verify_reason="returncode_zero" if ok else "nonzero_returncode",
            details={"returncode": completed.returncode},
        )
    except Exception as exc:  # pragma: no cover - system dependent
        return _result(False, error=str(exc), verified=False, verify_reason="exception")
