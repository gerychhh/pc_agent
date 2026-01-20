from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import yaml

from .config import PROJECT_ROOT, TIMEOUT_SEC
from .debug import debug_event, truncate_text
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
    score: int
    intent: str
    reason: str


@dataclass
class ExecResult:
    ok: bool
    stdout: str | None
    stderr: str | None
    returncode: int | None
    error: str | None = None


@dataclass
class CommandResult:
    action: Action
    execute_result: ExecResult
    verify_result: ExecResult | None
    ok: bool


def load_commands() -> list[dict[str, Any]]:
    if not COMMAND_LIBRARY_PATH.exists():
        return []
    data = yaml.safe_load(COMMAND_LIBRARY_PATH.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "commands" in data:
        return list(data["commands"] or [])
    if isinstance(data, list):
        return data
    return []


def match_command(user_text: str, commands: list[dict[str, Any]]) -> tuple[CommandMatch | None, list[CommandMatch]]:
    normalized = _normalize_text(user_text)
    matches: list[CommandMatch] = []
    for command in commands:
        for intent in command.get("intents") or []:
            intent_text = str(intent)
            score, reason = _match_intent(intent_text, normalized)
            if score == 0:
                continue
            params = extract_params(command, user_text, intent_text)
            matches.append(
                CommandMatch(
                    command=command,
                    params=params,
                    score=score,
                    intent=intent_text,
                    reason=reason,
                )
            )
    matches.sort(key=lambda m: m.score, reverse=True)
    best = matches[0] if matches else None
    debug_event("CMD_MATCH", f"matches={len(matches)} best={best.command.get('id') if best else 'NONE'}")
    for match in matches[:5]:
        debug_event(
            "CMD_MATCH",
            f"{match.command.get('id')} score={match.score} intent='{match.intent}' reason={match.reason}",
        )
    return best, matches


def render_template(script: str, params: dict[str, str]) -> str:
    rendered = script
    for key, value in params.items():
        rendered = rendered.replace(f"{{{{{key}}}}}", str(value))
    return rendered


def run_command(command: dict[str, Any], params: dict[str, str]) -> CommandResult:
    execute = command.get("execute") or {}
    verify = command.get("verify") or {}
    action = Action(
        language=execute.get("lang"),
        script=render_template(execute.get("script") or "", params),
        updates=command.get("state_update"),
        name=command.get("id"),
    )

    exec_result = _run_action(action)
    verify_result: ExecResult | None = None
    ok = exec_result.ok

    if verify:
        verify_action = Action(
            language=verify.get("lang"),
            script=render_template(verify.get("script") or "", params),
        )
        debug_event("VERIFY", f"script={truncate_text(verify_action.script, 200)}")
        verify_result = _run_action(verify_action)
        debug_event(
            "VERIFY",
            f"ok={verify_result.ok} stdout={truncate_text(verify_result.stdout or '', 200)} stderr={truncate_text(verify_result.stderr or '', 200)}",
        )
        ok = ok and verify_result.ok

    if ok and action.updates:
        _apply_state_updates(action.updates)

    return CommandResult(action=action, execute_result=exec_result, verify_result=verify_result, ok=ok)


def run_verify(verify: dict[str, Any], params: dict[str, str]) -> ExecResult:
    action = Action(language=verify.get("lang"), script=render_template(verify.get("script") or "", params))
    return _run_action(action)


def extract_params(command: dict[str, Any], user_text: str, intent_text: str) -> dict[str, str]:
    params: dict[str, str] = {}
    spec = command.get("params") or []
    wildcard_value = _extract_wildcard(intent_text, user_text)

    for item in spec:
        name = item.get("name")
        default = item.get("default")
        from_user = item.get("from_user", False)
        if from_user and name in {"text", "content"}:
            extracted = _extract_after_keywords(user_text)
            if extracted:
                params[name] = extracted
                continue
        if from_user and wildcard_value:
            params[name] = wildcard_value
        elif from_user:
            params[name] = _extract_after_keywords(user_text) or default or ""
        else:
            params[name] = default or ""

    if "query" in params:
        params["query"] = params["query"].strip()
    if "text" in params:
        params["text"] = params["text"].strip()
    if "content" in params:
        params["content"] = params["content"].strip()
    if "filename" in params:
        params["filename"] = _sanitize_filename(params["filename"], _guess_extension(command))

    return params


def _run_action(action: Action) -> ExecResult:
    if not action.language or not action.script:
        return ExecResult(ok=False, stdout="", stderr="empty action", returncode=1, error="invalid")
    debug_event("EXEC", f"lang={action.language} script={truncate_text(action.script, 200)}")
    validation = _validate_script(action.language, action.script)
    if validation:
        reason = "; ".join(validation)
        debug_event("VALIDATE", f"blocked: {reason}")
        return ExecResult(ok=False, stdout="", stderr=reason, returncode=1, error="blocked")
    debug_event("VALIDATE", "allowed")

    if action.language == "python":
        result = run_python(action.script, TIMEOUT_SEC)
    else:
        result = run_powershell(action.script, TIMEOUT_SEC)

    debug_event(
        "EXEC",
        f"returncode={result.get('returncode')} stdout={truncate_text(result.get('stdout') or '', 2000)} stderr={truncate_text(result.get('stderr') or '', 2000)}",
    )
    return ExecResult(
        ok=bool(result.get("ok")),
        stdout=result.get("stdout"),
        stderr=result.get("stderr"),
        returncode=result.get("returncode"),
    )


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


def _validate_script(language: str, script: str) -> list[str]:
    if language == "python":
        return validate_python(script)
    return validate_powershell(script)


def _normalize_text(text: str) -> str:
    return text.lower().strip()


def _match_intent(intent: str, text: str) -> tuple[int, str]:
    intent_lower = intent.lower().strip()
    if "*" in intent_lower:
        pattern = re.escape(intent_lower).replace("\\*", "(.+)")
        if re.search(pattern, text, re.IGNORECASE):
            return 3, "wildcard"
        return 0, ""
    if intent_lower in text:
        return 2, "substring"
    return 0, ""


def _extract_wildcard(intent: str, user_text: str) -> str | None:
    intent_lower = intent.lower().strip()
    if "*" not in intent_lower:
        return None
    pattern = re.escape(intent_lower).replace("\\*", "(.+)")
    match = re.search(pattern, user_text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _extract_after_keywords(text: str) -> str | None:
    match = re.search(r"(?:текст|содержимое|впиши|напиши)\s*[:\-]?\s*(.+)$", text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None


def _sanitize_filename(filename: str, extension: str | None) -> str:
    clean = re.sub(r"[^\w\-\.а-яА-Я]", "_", filename)
    if extension and not clean.lower().endswith(extension):
        clean = f"{clean}{extension}"
    return clean


def _guess_extension(command: dict[str, Any]) -> str | None:
    cmd_id = (command.get("id") or "").lower()
    if "txt" in cmd_id:
        return ".txt"
    if "docx" in cmd_id:
        return ".docx"
    return None
