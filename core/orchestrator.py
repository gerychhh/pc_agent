from __future__ import annotations

import hashlib
import os
import re
from typing import Any

from .config import DEBUG, FAST_MODEL, MAX_RETRIES, SMART_MODEL, TIMEOUT_SEC
from .executor import run_powershell, run_python
from .llm_client import LLMClient
from .logger import SessionLogger
from .policy import confirm_if_needed
from .router import route_task
from .skills import Action, match_skill
from .state import (
    add_recent_app,
    add_recent_file,
    add_recent_url,
    load_state,
    set_active_app,
    set_active_file,
    set_active_url,
)
from .validator import validate_powershell, validate_python


SYSTEM_PROMPT = """
Ты — локальный ассистент управления Windows-ПК через PowerShell и Python.
Отвечай кратко.
Если запрос содержит действие (открой/запусти/напиши/нажми/закрой и т.п.) — верни ровно один code block (powershell или python) и ничего больше.
Всегда возвращай РОВНО ОДИН fenced code block без дополнительного текста. Язык блока строго python или powershell.
Никогда не смешивай Python в PowerShell.
Никогда не выводи служебные блоки вроде [TOOL_RESULT].

SINGLE ACTION SCRIPT:
- Если запрос включает несколько шагов, сгенерируй ОДИН скрипт, который выполняет все шаги последовательно.
- Если нужны действия в UI (ввод текста, горячие клавиши, управление окном) — всегда используй Python (pyautogui + pygetwindow).
- Если нужно только открыть сайт/приложение без дальнейшего взаимодействия — PowerShell допустим.

ЭТАЛОННЫЙ ПАТТЕРН (Блокнот + ввод текста):
1) import time, subprocess, pyautogui, pygetwindow
2) subprocess.Popen(["notepad.exe"])
3) подожди 3-5 секунд, пока появится окно
4) найди окно и активируй его (Window.activate())
5) набери текст (pyautogui.typewrite(...))
6) опционально нажми Enter (pyautogui.press("enter"))
7) print("OK: <что сделано>")
ВАЖНО: нельзя печатать без активации окна. Нельзя писать “успешно” без фактического выполнения. Обязательно импортируй все используемое.

ФАЙЛЫ:
- .docx → только python-docx
- .xlsx → только openpyxl
- .pdf → только reportlab
- Нельзя создавать docx/xlsx/pdf через open(...,"w")

ACTIVE_FILE:
- Если ACTIVE_FILE задан и задача "измени/шрифт/формат", работай с ним.
- Для .docx открывай Document(active_file) и сохраняй в тот же файл.
""".strip()

REPORT_PROMPT = """
Ты — ассистент-репортёр выполнения команд на ПК.
Тебе дают: запрос пользователя, язык (python/powershell), выполненный скрипт, returncode, stdout, stderr, и состояние (active_file/url/app).
Напиши КОРОТКИЙ ответ пользователю на русском: что произошло.
- Если ok: 1-2 предложения, что сделано (например: "Открыл YouTube", "Создал файл на рабочем столе: ...")
- Если ошибка: коротко причина + 1 следующий шаг (без кода, если не просят)
Если есть строка "SKILL: youtube_video" — ответ должен быть: "Открыл видео на YouTube."
Никаких code block.
""".strip()


def sanitize_assistant_text(text: str) -> str:
    if not text:
        return ""
    cleaned = (
        text.replace("[TOOL_RESULT]", "")
        .replace("[END_TOOL_RESULT]", "")
        .replace("[/TOOL_RESULT]", "")
    )
    filtered_lines: list[str] = []
    for line in cleaned.splitlines():
        if any(marker in line for marker in ("<|channel|>", "<|constrain|>", "<|message|>")):
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines).strip()


def is_action_like(user_input: str) -> bool:
    verbs = (
        "открой",
        "закрой",
        "найди",
        "включи",
        "запусти",
        "создай",
        "измени",
        "сделай",
        "напиши",
        "введи",
        "нажми",
        "нажать",
        "клик",
        "кликни",
        "удали",
        "скачай",
        "перемотай",
        "пауза",
        "громче",
        "тише",
    )
    lowered = user_input.lower()
    return any(verb in lowered for verb in verbs)


class Orchestrator:
    def __init__(self, history_limit: int = 4) -> None:
        self.client = LLMClient()
        self.logger = SessionLogger()
        self.history_limit = history_limit
        self.system_message = {"role": "system", "content": SYSTEM_PROMPT}
        self._smart_banner_printed = False
        self.reset()

    def reset(self) -> None:
        self.history: list[dict[str, Any]] = []

    def _log_debug(self, label: str, value: str) -> None:
        if DEBUG:
            print(f"[{label}] {value}")

    @staticmethod
    def _script_hash(script: str) -> str:
        return hashlib.sha256(script.encode("utf-8")).hexdigest()[:8]

    @staticmethod
    def _extract_script(content: str) -> tuple[str | None, str | None]:
        if not content:
            return None, None
        stripped = content.strip()
        python_match = re.fullmatch(r"```python\s*([\s\S]*?)```", stripped, re.IGNORECASE)
        ps_match = re.fullmatch(r"```powershell\s*([\s\S]*?)```", stripped, re.IGNORECASE)
        if python_match and ps_match:
            return None, None
        if python_match:
            return "python", python_match.group(1).strip()
        if ps_match:
            return "powershell", ps_match.group(1).strip()
        if "[TOOL_RESULT]" in stripped:
            lang_match = re.search(r"LANG:\s*(python|powershell)", stripped, re.IGNORECASE)
            script_match = re.search(
                r"SCRIPT:\s*([\s\S]*?)(?:\n[A-Z_]+:|\[/TOOL_RESULT\]|$)",
                stripped,
                re.IGNORECASE,
            )
            script_block = script_match.group(1).strip() if script_match else ""
            if script_block:
                fenced_python = re.search(r"```python\s*([\s\S]*?)```", script_block, re.IGNORECASE)
                fenced_ps = re.search(r"```powershell\s*([\s\S]*?)```", script_block, re.IGNORECASE)
                if fenced_python:
                    return "python", fenced_python.group(1).strip()
                if fenced_ps:
                    return "powershell", fenced_ps.group(1).strip()
            if lang_match and script_block:
                return lang_match.group(1).lower(), script_block
        lines = stripped.splitlines()
        if lines:
            line_match = re.match(r"^(python|powershell)\s*$", lines[0], re.IGNORECASE)
        else:
            line_match = None
        if line_match:
            lang = line_match.group(1).lower()
            lines = lines[1:]
            script = "\n".join(lines).strip()
            if script:
                return lang, script
        command_match = re.match(
            r"^powershell\s+-Command\s+(.+)$", stripped, re.IGNORECASE | re.DOTALL
        )
        if command_match:
            payload = command_match.group(1).strip()
            if (payload.startswith('"') and payload.endswith('"')) or (
                payload.startswith("'") and payload.endswith("'")
            ):
                payload = payload[1:-1]
            payload = payload.strip()
            if payload:
                return "powershell", payload
        return None, None

    def _build_messages(
        self,
        user_input: str,
        stateless: bool,
        use_state: bool,
        state: dict[str, Any],
    ) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = [self.system_message]
        if use_state:
            context_lines = []
            if state.get("active_file"):
                context_lines.append(f"ACTIVE_FILE: {state['active_file']}")
            if state.get("active_url"):
                context_lines.append(f"ACTIVE_URL: {state['active_url']}")
            if state.get("active_app"):
                context_lines.append(f"ACTIVE_APP: {state['active_app']}")
            if context_lines:
                messages.append(
                    {
                        "role": "system",
                        "content": "[STATE]\n" + "\n".join(context_lines) + "\n[/STATE]",
                    }
                )
        if not stateless and self.history:
            messages.extend(self.history[-self.history_limit :])
        messages.append({"role": "user", "content": user_input})
        return messages

    def _execute_action(self, action: Action) -> dict[str, Any]:
        if action.language == "python":
            result = run_python(action.script, TIMEOUT_SEC)
        else:
            result = run_powershell(action.script, TIMEOUT_SEC)
        result = self._check_saved_output(result)
        saved_path = result.get("saved_path")
        if saved_path and result.get("ok"):
            # Если скрипт сообщил SAVED: <path>, сохраняем его как ACTIVE_FILE
            set_active_file(str(saved_path))
            add_recent_file(str(saved_path))
        return result


    def _update_state_from_action(self, action: Action) -> None:
        if not action.updates:
            return
        if "active_url" in action.updates:
            url = action.updates["active_url"]
            set_active_url(url)
            add_recent_url(url)
        if "active_app" in action.updates:
            app = action.updates["active_app"]
            set_active_app(app)
            add_recent_app(app)
        if "active_file" in action.updates:
            path = action.updates["active_file"]
            set_active_file(path)
            add_recent_file(path)

    def _validate_script(self, language: str, script: str) -> list[str]:
        if language == "python":
            return validate_python(script)
        return validate_powershell(script)

    def _run_llm(
        self,
        user_input: str,
        model_name: str,
        stateless: bool,
        use_state: bool,
        state: dict[str, Any],
    ) -> tuple[str | None, str | None, str]:
        messages = self._build_messages(user_input, stateless, use_state, state)
        self.logger.log_user_input(user_input, len(messages))
        try:
            response = self.client.chat(messages, tools=[], model_name=model_name, tool_choice="none")
            raw_content = response.choices[0].message.content or ""
        except Exception as exc:
            raw_content = f"LLM error: {exc}"
        self._log_debug("LLM_RAW", sanitize_assistant_text(raw_content)[:300])
        language, script = self._extract_script(raw_content)
        return language, script, raw_content

    def _run_report(self, payload: str, model_name: str) -> str:
        messages = [
            {"role": "system", "content": REPORT_PROMPT},
            {"role": "user", "content": payload},
        ]
        try:
            response = self.client.chat(messages, tools=[], model_name=model_name, tool_choice="none")
            content = response.choices[0].message.content or ""
            return sanitize_assistant_text(content)
        except Exception as exc:
            return f"LLM error: {exc}"

    def _print_smart_banner(self) -> None:
        if self._smart_banner_printed:
            return
        print("🧠 Сложная задача — подключаю умную модель...")
        self._smart_banner_printed = True

    @staticmethod
    def _check_saved_output(result: dict[str, Any]) -> dict[str, Any]:
        stdout = result.get("stdout") or ""
        match = re.search(r"^(?:SAVED|OK):\\s*(.+)$", stdout, re.MULTILINE)
        if not match:
            return result
        path = match.group(1).strip()
        if os.path.exists(path):
            updated = dict(result)
            updated["saved_path"] = path
            return updated
        updated = dict(result)
        updated["ok"] = False
        stderr = (result.get("stderr") or "").strip()
        missing_msg = f"Файл не найден по пути: {path}"
        updated["stderr"] = f"{stderr}\n{missing_msg}".strip()
        updated["returncode"] = result.get("returncode") or 1
        return updated

    def _build_report_payload(
        self,
        user_input: str,
        language: str,
        script: str,
        result: dict[str, Any],
        action: Action | None = None,
    ) -> str:
        state = load_state()
        active_file = state.get("active_file") or ""
        active_url = state.get("active_url") or ""
        active_app = state.get("active_app") or ""
        lines = [
            f"USER: {user_input}",
            f"LANG: {language}",
            f"SCRIPT: {script}",
            f"RESULT_OK: {str(result.get('ok', False)).lower()}",
            f"RETURNCODE: {result.get('returncode')}",
            f"STDOUT: {result.get('stdout') or ''}",
            f"STDERR: {result.get('stderr') or ''}",
            f"STATE: ACTIVE_FILE={active_file} | ACTIVE_URL={active_url} | ACTIVE_APP={active_app}",
        ]
        if action and action.name:
            lines.append(f"SKILL: {action.name}")
        return "\n".join(lines)

    def _build_tool_result_block(
        self, language: str, script: str, result: dict[str, Any]
    ) -> str:
        return (
            "[TOOL_RESULT]\n"
            f"LANG: {language}\n"
            f"SCRIPT: {script}\n"
            f"RESULT_OK: {str(result.get('ok', False)).lower()}\n"
            f"RETURNCODE: {result.get('returncode')}\n"
            f"STDOUT: {result.get('stdout') or ''}\n"
            f"STDERR: {result.get('stderr') or ''}\n"
            "[/TOOL_RESULT]"
        )

    def _report_action(
        self,
        user_input: str,
        language: str,
        script: str,
        result: dict[str, Any],
        *,
        action: Action | None = None,
        complex_task: bool = False,
    ) -> str:
        report_model = SMART_MODEL if (not result.get("ok") or complex_task) else FAST_MODEL
        if report_model == SMART_MODEL:
            self._print_smart_banner()
        payload = self._build_report_payload(user_input, language, script, result, action=action)
        return self._run_report(payload, report_model)

    def _confirm_action(self, action: Action) -> bool:
        return confirm_if_needed(action.language, action.script)

    def run(self, user_input: str, stateless: bool = False) -> str:
        self._smart_banner_printed = False
        state = load_state()
        skill_result = match_skill(user_input, state)
        if isinstance(skill_result, str):
            response = skill_result
            if not stateless:
                self._store_history(user_input, response)
            return response
        if isinstance(skill_result, Action):
            if not self._confirm_action(skill_result):
                response = "Отменено пользователем."
                if not stateless:
                    self._store_history(user_input, response)
                return response
            result = self._execute_action(skill_result)
            if result.get("ok"):
                self._update_state_from_action(skill_result)
            response = self._report_action(
                user_input,
                skill_result.language,
                skill_result.script,
                result,
                action=skill_result,
            )
            if not stateless:
                tool_result = self._build_tool_result_block(
                    skill_result.language,
                    skill_result.script,
                    result,
                )
                self._store_history(user_input, response, tool_result=tool_result)
            return response

        route = route_task(user_input)
        complexity = route["complexity"]
        force_lang = route["force_lang"]
        model_name = FAST_MODEL if complexity == "simple" else SMART_MODEL
        use_state = model_name == SMART_MODEL
        if model_name == SMART_MODEL:
            self._print_smart_banner()

        language, script, raw_content = self._run_llm(
            user_input,
            model_name,
            stateless=stateless or model_name == FAST_MODEL,
            use_state=use_state,
            state=state,
        )
        if raw_content.startswith("LLM error:"):
            response = raw_content
            if not stateless:
                self._store_history(user_input, response)
            return response
        if not language or not script:
            format_prompt = (
                "Ты вернул ответ без code block. Верни то же самое, но строго в одном "
                "fenced code block (python или powershell) и без текста."
            )
            language, script, raw_content = self._run_llm(
                f"{user_input}\n{format_prompt}",
                model_name,
                stateless=True,
                use_state=use_state,
                state=state,
            )
            if raw_content.startswith("LLM error:"):
                response = raw_content
                if not stateless:
                    self._store_history(user_input, response)
                return response
            if not language or not script:
                self._print_smart_banner()
                language, script, raw_content = self._run_llm(
                    f"{user_input}\n{format_prompt}",
                    SMART_MODEL,
                    stateless=False,
                    use_state=True,
                    state=state,
                )
            if raw_content.startswith("LLM error:"):
                response = raw_content
                if not stateless:
                    self._store_history(user_input, response)
                return response
            if not language or not script:
                if is_action_like(user_input):
                    response = "Не смог сформировать команду для выполнения."
                else:
                    response = sanitize_assistant_text(raw_content)
                if not stateless:
                    self._store_history(user_input, response)
                return response
        if force_lang and language != force_lang:
            response, tool_result = self._escalate_with_errors(
                user_input,
                script,
                [f"Нужен язык {force_lang}, но получен {language}."],
                state,
            )
            if not stateless:
                self._store_history(user_input, response, tool_result=tool_result)
            return response

        errors = self._validate_script(language, script)
        if errors:
            response, tool_result = self._escalate_with_errors(user_input, script, errors, state)
            if not stateless:
                self._store_history(user_input, response, tool_result=tool_result)
            return response

        action = Action(language=language, script=script)
        if not self._confirm_action(action):
            response = "Отменено пользователем."
            if not stateless:
                self._store_history(user_input, response)
            return response
        result = self._execute_action(action)
        if result.get("ok"):
            self._maybe_track_success(language, script)
            response = self._report_action(
                user_input,
                language,
                script,
                result,
                complex_task=complexity == "complex",
            )
            if not stateless:
                tool_result = self._build_tool_result_block(language, script, result)
                self._store_history(user_input, response, tool_result=tool_result)
            return response

        stdout = result.get("stdout") or ""
        stderr = result.get("stderr") or ""
        retry_outcome = self._retry_with_smart(user_input, language, script, stdout, stderr, state)
        if retry_outcome is None:
            response = "Отменено пользователем."
            if not stateless:
                self._store_history(user_input, response)
            return response
        retry_language, retry_script, retry_result = retry_outcome
        response = self._report_action(
            user_input,
            retry_language,
            retry_script,
            retry_result,
            complex_task=True,
        )
        if not stateless:
            tool_result = self._build_tool_result_block(retry_language, retry_script, retry_result)
            self._store_history(user_input, response, tool_result=tool_result)
        return response

    def _store_history(self, user_input: str, response: str, tool_result: str | None = None) -> None:
        entries = [{"role": "user", "content": user_input}]
        if tool_result:
            entries.append({"role": "assistant", "content": tool_result})
        entries.append({"role": "assistant", "content": response})
        self.history.extend(entries)
        self.history = self.history[-self.history_limit :]

    def _maybe_track_success(self, language: str, script: str) -> None:
        if language != "powershell":
            return
        match = re.search(r"Start-Process\s+['\"](https?://[^'\"]+)['\"]", script, re.IGNORECASE)
        if match:
            url = match.group(1)
            set_active_url(url)
            add_recent_url(url)
            return
        app_match = re.search(r"Start-Process\s+([\\w.\\\\-]+)", script, re.IGNORECASE)
        if app_match:
            app = app_match.group(1)
            set_active_app(app)
            add_recent_app(app)

    def _escalate_with_errors(
        self, user_input: str, script: str, errors: list[str], state: dict[str, Any]
    ) -> tuple[str, str | None]:
        error_text = "\n".join(f"- {err}" for err in errors)
        prompt = (
            "Скрипт не прошёл валидацию:\n"
            f"{error_text}\n"
            "Исправь и верни только один корректный code block."
        )
        self._print_smart_banner()
        language, new_script, _ = self._run_llm(
            f"{user_input}\n{prompt}",
            SMART_MODEL,
            stateless=False,
            use_state=True,
            state=state,
        )
        if not language or not new_script:
            return "Не удалось получить корректный скрипт.", None
        errors = self._validate_script(language, new_script)
        if errors:
            return "Скрипт все еще невалиден.", None
        action = Action(language=language, script=new_script)
        if not self._confirm_action(action):
            return "Отменено пользователем.", None
        result = self._execute_action(action)
        if result.get("ok"):
            self._maybe_track_success(language, new_script)
        response = self._report_action(
            user_input,
            language,
            new_script,
            result,
            complex_task=True,
        )
        tool_result = self._build_tool_result_block(language, new_script, result)
        return response, tool_result

    def _retry_with_smart(
        self,
        user_input: str,
        language: str,
        script: str,
        stdout: str,
        stderr: str,
        state: dict[str, Any],
    ) -> tuple[str, str, dict[str, Any]] | None:
        self._print_smart_banner()
        last_language = language
        last_script = script
        last_result = {"ok": False, "stdout": stdout, "stderr": stderr, "returncode": None}
        for attempt in range(MAX_RETRIES):
            print("🔁 Ошибка выполнения — пробую исправить...")
            script_to_fix = last_script
            prompt = (
                "Скрипт упал. Исправь и верни только новый корректный code block.\n"
                f"USER:\n{user_input}\n"
                f"SCRIPT:\n{script_to_fix}\n"
                f"STDOUT:\n{stdout}\n"
                f"STDERR:\n{stderr}\n"
            )
            language, new_script, _ = self._run_llm(
                prompt,
                SMART_MODEL,
                stateless=False,
                use_state=True,
                state=state,
            )
            if not language or not new_script:
                continue
            errors = self._validate_script(language, new_script)
            if errors:
                stdout = ""
                stderr = "\n".join(errors)
                last_language = language
                last_script = new_script
                last_result = {
                    "ok": False,
                    "stdout": stdout,
                    "stderr": stderr,
                    "returncode": None,
                }
                continue
            action = Action(language=language, script=new_script)
            if not self._confirm_action(action):
                return None
            result = self._execute_action(action)
            if result.get("ok"):
                self._maybe_track_success(language, new_script)
                return language, new_script, result
            stdout = result.get("stdout") or ""
            stderr = result.get("stderr") or ""
            last_language = language
            last_script = new_script
            last_result = result
        return last_language, last_script, last_result
