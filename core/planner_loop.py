from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .command_engine import Action, extract_params, load_commands, run_command
from .context_builder import ctx_planner, ctx_smart_fix
from .debug import debug_event, truncate_text
from .llm_client import LLMClient
from .llm_parser import parse_step_json
from .validator import validate_powershell, validate_python
from .config import SMART_MODEL, TIMEOUT_SEC
from .executor import run_powershell, run_python


@dataclass
class StepExecution:
    step: dict[str, Any]
    action: Action | None
    execute_result: dict[str, Any] | None
    verify_result: dict[str, Any] | None
    ok: bool
    blocked: bool


class PlannerLoop:
    def __init__(self, max_steps: int = 6, max_fix_attempts: int = 3) -> None:
        self.client = LLMClient()
        self.max_steps = max_steps
        self.max_fix_attempts = max_fix_attempts
        self.commands = load_commands()

    def run(self, user_text: str, state: dict[str, Any], command_index: str) -> list[StepExecution]:
        results: list[StepExecution] = []
        for step_num in range(1, self.max_steps + 1):
            step = self._plan_step(user_text, state, command_index, results)
            if not step:
                break
            if step.get("execute") is None and step.get("verify") is None:
                debug_event("PLANNER", "planner returned done")
                break
            debug_event("PLANNER", f"step {step_num} id={step.get('step_id')}")
            execution = self._execute_step(step, user_text)
            results.append(execution)
            debug_event("PLANNER", f"step {step_num} ok={execution.ok} blocked={execution.blocked}")
            if execution.blocked:
                break
            if not execution.ok and step.get("stop_if_failed", True):
                fixed = self._smart_fix_step(user_text, state, command_index, step, execution)
                if fixed:
                    results.append(fixed)
                    debug_event("SMART_FIX", f"fixed ok={fixed.ok} blocked={fixed.blocked}")
                    if fixed.blocked or not fixed.ok:
                        break
                else:
                    break
        return results

    def _plan_step(
        self,
        user_text: str,
        state: dict[str, Any],
        command_index: str,
        results: list[StepExecution],
    ) -> dict[str, Any] | None:
        last_run = [
            {
                "step_id": item.step.get("step_id"),
                "ok": item.ok,
                "stderr": truncate_text((item.execute_result or {}).get("stderr") or "", 200),
            }
            for item in results
        ]
        payload = ctx_planner(state, command_index, user_text, last_run=last_run)
        for attempt in range(2):
            response = self.client.chat(
                [{"role": "user", "content": payload}],
                tools=[],
                model_name=SMART_MODEL,
                tool_choice="none",
            )
            content = response.choices[0].message.content or ""
            step = parse_step_json(content)
            if step:
                return step
            payload = payload + "\n[RETRY] Верни только JSON по формату, без текста и без code blocks."
            debug_event("PARSE", f"planner retry {attempt + 1}")
        return None

    def _execute_step(self, step: dict[str, Any], user_text: str) -> StepExecution:
        use_command = step.get("use_command_id")
        if use_command:
            command = next((cmd for cmd in self.commands if cmd.get("id") == use_command), None)
            if command:
                params = extract_params(command, user_text, "")
                result = run_command(command, params)
                return StepExecution(
                    step=step,
                    action=result.action,
                    execute_result=_as_dict(result.execute_result),
                    verify_result=_as_dict(result.verify_result) if result.verify_result else None,
                    ok=result.ok,
                    blocked=_is_blocked(result.execute_result, result.verify_result),
                )

        execute = step.get("execute") or {}
        verify = step.get("verify") or {}
        action = _action_from_step(execute)
        if not action:
            return StepExecution(step=step, action=None, execute_result=None, verify_result=None, ok=False, blocked=False)
        errors = _validate_action(action)
        if errors:
            return StepExecution(
                step=step,
                action=action,
                execute_result={"ok": False, "stderr": "\n".join(errors), "returncode": 1, "error": "blocked"},
                verify_result=None,
                ok=False,
                blocked=True,
            )
        exec_result = _run_action(action)
        ok = exec_result.get("ok", False)
        verify_result = None
        if verify:
            verify_action = _action_from_step(verify)
            if verify_action:
                debug_event("VERIFY", f"script={truncate_text(verify_action.script, 200)}")
                verify_errors = _validate_action(verify_action)
                if verify_errors:
                    verify_result = {"ok": False, "stderr": "\n".join(verify_errors), "returncode": 1, "error": "blocked"}
                    ok = False
                else:
                    verify_result = _run_action(verify_action)
                    ok = ok and verify_result.get("ok", False)
        return StepExecution(
            step=step,
            action=action,
            execute_result=exec_result,
            verify_result=verify_result,
            ok=ok,
            blocked=False,
        )

    def _smart_fix_step(
        self,
        user_text: str,
        state: dict[str, Any],
        command_index: str,
        step: dict[str, Any],
        execution: StepExecution,
    ) -> StepExecution | None:
        last_result = execution.execute_result or {"ok": False, "stdout": "", "stderr": ""}
        for attempt in range(1, self.max_fix_attempts + 1):
            debug_event("SMART_FIX", f"attempt {attempt}")
            payload = ctx_smart_fix(state, command_index, step, last_result, user_text)
            response = self.client.chat(
                [{"role": "user", "content": payload}],
                tools=[],
                model_name=SMART_MODEL,
                tool_choice="none",
            )
            content = response.choices[0].message.content or ""
            fixed_step = parse_step_json(content)
            if not fixed_step:
                continue
            fixed_exec = self._execute_step(fixed_step, user_text)
            if fixed_exec.blocked:
                return fixed_exec
            if fixed_exec.ok:
                return fixed_exec
            last_result = fixed_exec.execute_result or last_result
        return None


def _action_from_step(step_payload: dict[str, Any]) -> Action | None:
    language = step_payload.get("lang")
    script = step_payload.get("script")
    if not language or not script:
        return None
    return Action(language=language, script=script)


def _validate_action(action: Action) -> list[str]:
    if action.language == "python":
        errors = validate_python(action.script)
    else:
        errors = validate_powershell(action.script)
    if errors:
        debug_event("VALIDATE", f"blocked: {'; '.join(errors)}")
    else:
        debug_event("VALIDATE", "allowed")
    return errors


def _run_action(action: Action) -> dict[str, Any]:
    debug_event("EXEC", f"lang={action.language} script={truncate_text(action.script, 200)}")
    if action.language == "python":
        result = run_python(action.script, TIMEOUT_SEC)
    else:
        result = run_powershell(action.script, TIMEOUT_SEC)
    debug_event(
        "EXEC",
        f"returncode={result.get('returncode')} stdout={truncate_text(result.get('stdout') or '', 2000)} stderr={truncate_text(result.get('stderr') or '', 2000)}",
    )
    return result


def _as_dict(result: Any) -> dict[str, Any]:
    if result is None:
        return {}
    if isinstance(result, dict):
        return result
    return {
        "ok": result.ok,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "returncode": result.returncode,
        "error": result.error,
    }


def _is_blocked(execute_result: Any, verify_result: Any | None) -> bool:
    for result in (execute_result, verify_result or {}):
        if isinstance(result, dict) and result.get("error") == "blocked":
            return True
    return False
