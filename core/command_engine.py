from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from urllib.parse import quote_plus

import yaml

from .config import PROJECT_ROOT, TIMEOUT_SEC
from .executor import run_powershell, run_python
from .state import add_recent_app, add_recent_file, add_recent_url, set_active_app, set_active_file, set_active_url
from .validator import validate_powershell, validate_python


COMMAND_LIBRARY_PATH = PROJECT_ROOT / "core" / "command_library.yaml"


@dataclass
class Action:
    language: str
    script: str
    updates: dict[str, Any] | None = None
    name: str | None = None


@dataclass
class CommandMatch:
    command: dict[str, Any]
    params: dict[str, str]
    missing: list[str]
    score: int


@dataclass
class CommandResult:
    action: Action
    execute_result: dict[str, Any]
    verify_result: dict[str, Any] | None
    ok: bool


def load_commands() -> list[dict[str, Any]]:
    if not COMMAND_LIBRARY_PATH.exists():
        return []
    data = yaml.safe_load(COMMAND_LIBRARY_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        return []
    return data


def _normalize_text(text: str) -> str:
    lowered = text.lower()
    replacements = {
        "ткст": "txt",
        "тхт": "txt",
        "ютуб": "youtube",
    }
    for old, new in replacements.items():
        lowered = lowered.replace(old, new)
    return lowered


def _intent_score(intent: str, text: str) -> int:
    cleaned = intent.strip()
    if cleaned.startswith("re:"):
        pattern = cleaned[3:].strip()
        return 3 if re.search(pattern, text, re.IGNORECASE) else 0
    if cleaned.startswith("/") and cleaned.endswith("/") and len(cleaned) > 2:
        pattern = cleaned[1:-1]
        return 3 if re.search(pattern, text, re.IGNORECASE) else 0
    return 2 if cleaned in text else 0


def match_command(user_text: str, commands: list[dict[str, Any]]) -> CommandMatch | None:
    normalized = _normalize_text(user_text)
    best: CommandMatch | None = None
    for command in commands:
        intents = command.get("intents") or []
        score = 0
        for intent in intents:
            score = max(score, _intent_score(str(intent).lower(), normalized))
        if score == 0:
            continue
        params, missing = extract_params(command, user_text)
        match = CommandMatch(command=command, params=params, missing=missing, score=score)
        if best is None or match.score > best.score:
            best = match
    return best


def _required_placeholders(command: dict[str, Any]) -> set[str]:
    placeholders: set[str] = set()
    for key in ("execute", "verify"):
        section = command.get(key) or {}
        script = section.get("script") or ""
        for name in re.findall(r"{(\w+)}", script):
            placeholders.add(name)
    return placeholders


def extract_params(command: dict[str, Any], user_text: str) -> tuple[dict[str, str], list[str]]:
    required = _required_placeholders(command)
    params: dict[str, str] = {}
    lowered = user_text.lower()

    if "query" in required:
        match = re.search(r"найди на (?:ютубе|youtube|ютуб)\s+(.+)", user_text, re.IGNORECASE)
        if match:
            params["query"] = match.group(1).strip()

    if "content" in required:
        content_match = re.search(
            r"(?:впиши\s+туда|напиши\s+туда|текст\s*:|содержимое\s*:|и\s+впиши)\s*(.+)$",
            user_text,
            re.IGNORECASE,
        )
        if content_match:
            params["content"] = content_match.group(1).strip()
        elif command.get("id", "").endswith("DOCX_DESKTOP"):
            params["content"] = "Я — локальный ассистент управления Windows‑ПК через PowerShell и Python."

    if "filename" in required:
        name_match = re.search(r"(\S+\.(?:txt|docx))", user_text, re.IGNORECASE)
        if name_match:
            params["filename"] = name_match.group(1).strip()
        elif "txt" in lowered:
            params["filename"] = "note.txt"
        elif "docx" in lowered or "ворд" in lowered or "word" in lowered:
            params["filename"] = "about.docx"

    missing = [name for name in required if not params.get(name)]
    return params, missing


def _sanitize_filename(filename: str, extension: str) -> str:
    clean = re.sub(r"[^\w\-\.а-яА-Я]", "_", filename)
    if not clean.lower().endswith(extension):
        clean = f"{clean}{extension}"
    return clean


def _render_script(script: str, language: str, params: dict[str, str]) -> str:
    rendered = script
    for key, value in params.items():
        if key == "query":
            safe_value = quote_plus(value)
        else:
            safe_value = value
        if language == "python":
            rendered = rendered.replace(f"{{{key}}}", repr(safe_value))
        else:
            if key == "content":
                safe_value = safe_value.replace("@'", "@`'")
            rendered = rendered.replace(f"{{{key}}}", str(safe_value))
    return rendered


def render_command(match: CommandMatch) -> Action:
    command = match.command
    execute = command.get("execute") or {}
    language = execute.get("lang")
    script = execute.get("script") or ""

    params = dict(match.params)
    if "filename" in params:
        if command["id"].endswith("TXT_DESKTOP"):
            params["filename"] = _sanitize_filename(params["filename"], ".txt")
        if command["id"].endswith("DOCX_DESKTOP"):
            params["filename"] = _sanitize_filename(params["filename"], ".docx")
    if "content" in params and command["id"].endswith("DOCX_DESKTOP"):
        if not params["content"]:
            params["content"] = "Я — локальный ассистент управления Windows‑ПК через PowerShell и Python."

    script = _render_script(script, language, params)
    updates = command.get("state_update")
    return Action(language=language, script=script, updates=updates, name=command.get("id"))


def _render_verify(command: dict[str, Any], params: dict[str, str]) -> tuple[str, str] | None:
    verify = command.get("verify") or {}
    language = verify.get("lang")
    script = verify.get("script")
    if not language or not script:
        return None
    rendered = _render_script(script, language, params)
    return language, rendered


def _validate(language: str, script: str) -> list[str]:
    if language == "python":
        return validate_python(script)
    return validate_powershell(script)


def run_command(match: CommandMatch) -> CommandResult:
    command = match.command
    action = render_command(match)
    execute_errors = _validate(action.language, action.script)
    if execute_errors:
        return CommandResult(
            action=action,
            execute_result={
                "ok": False,
                "stderr": "\n".join(execute_errors),
                "returncode": 1,
                "error": "blocked",
            },
            verify_result=None,
            ok=False,
        )

    execute_result = run_python(action.script, TIMEOUT_SEC) if action.language == "python" else run_powershell(action.script, TIMEOUT_SEC)
    verify_result: dict[str, Any] | None = None
    ok = execute_result.get("ok", False)

    verify_payload = _render_verify(command, match.params)
    if verify_payload:
        verify_lang, verify_script = verify_payload
        verify_errors = _validate(verify_lang, verify_script)
        if verify_errors:
            verify_result = {
                "ok": False,
                "stderr": "\n".join(verify_errors),
                "returncode": 1,
                "error": "blocked",
            }
            ok = False
        else:
            verify_result = (
                run_python(verify_script, TIMEOUT_SEC)
                if verify_lang == "python"
                else run_powershell(verify_script, TIMEOUT_SEC)
            )
            ok = ok and verify_result.get("ok", False)

    if ok and action.updates:
        _apply_state_updates(action.updates)

    return CommandResult(action=action, execute_result=execute_result, verify_result=verify_result, ok=ok)


def _apply_state_updates(updates: dict[str, Any]) -> None:
    if "active_url" in updates:
        url = updates["active_url"]
        set_active_url(url)
        add_recent_url(url)
    if "active_app" in updates:
        app = updates["active_app"]
        set_active_app(app)
        add_recent_app(app)
    if "active_file" in updates:
        path = updates["active_file"]
        set_active_file(path)
        add_recent_file(path)
