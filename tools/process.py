from __future__ import annotations

import json
import os
import re
from typing import Any
from urllib.parse import urlparse

from tools import commands


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


def _looks_like_path(app: str) -> bool:
    return any(sep in app for sep in ("/", "\\")) or app.lower().endswith(".exe")


def _clean_query_name(text: str) -> str:
    cleaned = (text or "").strip().strip('"').strip("'")
    if not cleaned:
        return ""
    if "\\" in cleaned or "/" in cleaned:
        cleaned = os.path.basename(cleaned)
    if cleaned.lower().endswith(".exe"):
        cleaned = cleaned[:-4]
    return cleaned.strip()


def _looks_like_url(text: str) -> bool:
    if not text:
        return False
    if re.match(r"^https?://", text, re.IGNORECASE):
        return True
    parsed = urlparse(text)
    if parsed.scheme and parsed.netloc:
        return True
    return bool(re.search(r"\bwww\.[^\s]+", text, re.IGNORECASE))


def _verify_process(process_name: str) -> dict[str, Any]:
    if not process_name:
        return {
            "verified": False,
            "verify_reason": "missing_process_name",
            "verify_details": {},
            "verify_exec": None,
        }
    ps_command = (
        f"Get-Process -Name \"{process_name}\" -ErrorAction SilentlyContinue | Select-Object -First 1"
    )
    raw = commands.run_powershell(ps_command)
    parsed = json.loads(raw)
    stdout = (parsed.get("stdout") or "").strip()
    verified = bool(stdout)
    return {
        "verified": verified,
        "verify_reason": "process_detected" if verified else "process_not_found",
        "verify_details": {"process_name": process_name},
        "verify_exec": parsed.get("details", {}).get("exec"),
    }


def open_app(app: str, alias: str | None = None) -> str:
    target = app
    clean_name = _clean_query_name(app)

    if _looks_like_url(app):
        return _result(
            False,
            app=alias or app,
            error="use_open_url",
            method="start-process",
            target=target,
            verified=False,
            verify_reason="url_detected",
        )

    launch_result: dict[str, Any] | None = None
    method = "start-process"

    if _looks_like_path(app) and app.lower().endswith(".exe") and os.path.exists(app):
        launch_raw = commands.run_powershell(f'Start-Process -FilePath "{app}"')
        launch_result = json.loads(launch_raw)
        method = "path"
    else:
        launch_raw = commands.run_powershell(f'Start-Process "{app}"')
        launch_result = json.loads(launch_raw)
        if not launch_result.get("ok"):
            query = clean_name or app
            ps = (
                "$q=\"" + query.replace('"', '') + "\"; "
                "Get-StartApps | Where-Object { $_.Name -like ('*'+$q+'*') } "
                "| Select-Object -First 1 Name, AppID | ConvertTo-Json -Compress"
            )
            search_raw = commands.run_powershell(ps)
            search_parsed = json.loads(search_raw)
            try:
                app_entry = json.loads((search_parsed.get("stdout") or "").strip() or "null")
            except json.JSONDecodeError:
                app_entry = None
            if app_entry and app_entry.get("AppID"):
                app_id = app_entry["AppID"]
                launch_raw = commands.run_cmd(f'explorer.exe "shell:AppsFolder\\{app_id}"')
                launch_result = json.loads(launch_raw)
                method = "appsfolder"
                target = app_id

    if not launch_result:
        return _result(
            False,
            app=alias or app,
            error="launch_failed",
            method=method,
            target=target,
            verified=False,
            verify_reason="launch_not_attempted",
        )

    ok = bool(launch_result.get("ok"))
    verify = _verify_process(clean_name)

    return _result(
        ok,
        app=alias or app,
        error=None if ok else (launch_result.get("stderr") or "launch_failed"),
        method=method,
        target=target,
        verified=verify["verified"],
        verify_reason=verify["verify_reason"],
        verify_details=verify["verify_details"],
        stdout=launch_result.get("stdout"),
        stderr=launch_result.get("stderr"),
        duration_ms=launch_result.get("duration_ms"),
        details={
            "exec": launch_result.get("details", {}).get("exec"),
            "verify_exec": verify.get("verify_exec"),
        },
    )


def open_url(url: str) -> str:
    launch_raw = commands.run_powershell(f'Start-Process "{url}"')
    launch_result = json.loads(launch_raw)
    ok = bool(launch_result.get("ok"))
    return _result(
        ok,
        url=url,
        done=True,
        method="start-process",
        verified=ok,
        verify_reason="returncode_zero" if ok else "nonzero_returncode",
        stdout=launch_result.get("stdout"),
        stderr=launch_result.get("stderr"),
        duration_ms=launch_result.get("duration_ms"),
        details={"exec": launch_result.get("details", {}).get("exec")},
    )
