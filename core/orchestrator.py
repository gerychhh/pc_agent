from __future__ import annotations

import hashlib
import re
from typing import Any

from .config import DEBUG, MAX_RETRIES, TIMEOUT_SEC
from .executor import run_pip_install, run_powershell, run_python
from .llm_client import LLMClient
from .logger import SessionLogger
from .policy import RiskLevel, assess_risk, confirm_if_needed


SYSTEM_PROMPT = """
Ты — локальный ассистент управления Windows-ПК через PowerShell и Python.

Отвечай КРАТКО.

ГЛАВНОЕ ПРАВИЛО:
Если пользователь просит СДЕЛАТЬ действие на компьютере (открыть/запустить/перейти/создать/скачать/найти/настроить/закрыть) —
ТЫ ОБЯЗАН вернуть КОД ДЛЯ ВЫПОЛНЕНИЯ.
Не спорь, не объясняй, не “я не это”, а просто делай.

ФОРМАТ ОТВЕТА:
1) Если действие НЕ нужно (вопрос/объяснение) — 1–3 предложения, БЕЗ кода.
2) Если действие НУЖНО — верни РОВНО ОДИН блок кода и НИЧЕГО больше:
   - либо ```powershell ... ```
   - либо ```python ... ```
Вне code block не пиши ни одного символа.

ПРАВИЛА ДЛЯ ДЕЙСТВИЙ:
- По умолчанию используй PowerShell.
- Скрипт должен быть ПОЛНЫМ и ИСПОЛНЯЕМЫМ.
- Если нужно открыть сайт — используй:
  Start-Process "https://...."
- Если пользователь пишет "ютуб / youtube" — открывай https://www.youtube.com/

ОКРУЖЕНИЕ:
- ОС: Windows, shell: PowerShell.
- Python доступен как `python`.
- Для Desktop:
  - PowerShell: `$desktop = [Environment]::GetFolderPath('Desktop')`
  - Python: `from pathlib import Path; desktop = Path.home() / "Desktop"`
  
ФАЙЛЫ И ФОРМАТЫ (ОЧЕНЬ ВАЖНО):
Если пользователь просит создать файл определённого типа (docx, xlsx, pdf, png, mp3 и т.д.) —
файл должен быть РЕАЛЬНОГО ФОРМАТА, а не просто текстом с таким расширением.

Запрещено делать так:
- open("file.docx","w").write("...")  # это НЕ docx
- open("file.pdf","w").write("...")   # это НЕ pdf

Делай правильно:
- .docx → только через библиотеку python-docx:
  from docx import Document; doc = Document(); doc.add_paragraph(...); doc.save("file.docx")
- .xlsx → только через openpyxl или pandas ExcelWriter
- .pdf  → только через reportlab (или другой генератор PDF)
- .json/.txt/.md/.csv → можно через обычный open(...,"w") (это текстовые форматы)
- картинки (.png/.jpg) → генерируй/сохраняй как изображение, не текст

ВЫБОР ЯЗЫКА ДЛЯ ФАЙЛОВ:
- .docx / .xlsx / .pdf → всегда Python (правильные библиотеки)
- открыть сайт / запустить программу / команды ОС → PowerShell

Если библиотека недоступна или формат неясен — задай 1 уточняющий вопрос или предложи правильный способ.

ВАЖНО ПРО ЯЗЫК СКРИПТА:
Если в решении используется Python-код или Python-библиотеки (например python-docx, openpyxl, reportlab и т.д.) —
ТОГДА скрипт обязан быть в формате ```python``` и выполняться как Python.
Никогда не вставляй Python-код внутрь PowerShell.

PowerShell используй только для команд Windows (Start-Process, cd, dir, New-Item и т.д.).

БЕЗОПАСНОСТЬ:
- Не делай разрушительные действия.
- Если запрос опасный или неясный — задай 1 короткий уточняющий вопрос.
""".strip()




def sanitize_assistant_text(text: str) -> str:
    if not text:
        return ""
    cleaned = text.replace("[TOOL_RESULT]", "").replace("[END_TOOL_RESULT]", "")
    filtered_lines: list[str] = []
    for line in cleaned.splitlines():
        if any(marker in line for marker in ("<|channel|>", "<|constrain|>", "<|message|>")):
            continue
        filtered_lines.append(line)
    return "\n".join(filtered_lines).strip()


class Orchestrator:
    def __init__(self) -> None:
        self.client = LLMClient()
        self.logger = SessionLogger()
        self.reset()

    def reset(self) -> None:
        self.messages: list[dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    @staticmethod
    def _extract_script(content: str) -> tuple[str | None, str | None]:
        if not content:
            return None, None
        python_match = re.search(r"```python\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
        if python_match:
            return "python", python_match.group(1).strip()
        ps_match = re.search(r"```powershell\s*(.*?)```", content, re.DOTALL | re.IGNORECASE)
        if ps_match:
            return "powershell", ps_match.group(1).strip()
        return None, None

    @staticmethod
    def _is_command_text(content: str) -> bool:
        if not content:
            return False
        trimmed = content.strip()
        command_prefixes = (
            "get-",
            "set-",
            "start-",
            "new-",
            "remove-",
            "copy-",
            "move-",
            "invoke-",
            "add-",
            "python ",
            "pip ",
            "dir",
            "cd ",
            "echo ",
            "notepad",
            "start ",
        )
        lowered = trimmed.lower()
        return lowered.startswith(command_prefixes)

    def _run_script(self, language: str, script: str) -> dict[str, Any]:
        if language == "python":
            return run_python(script, TIMEOUT_SEC)
        return run_powershell(script, TIMEOUT_SEC)

    def _log_debug(self, label: str, value: str) -> None:
        if DEBUG:
            print(f"[{label}] {value}")

    @staticmethod
    def _script_hash(script: str) -> str:
        return hashlib.sha256(script.encode("utf-8")).hexdigest()[:8]

    @staticmethod
    def _extract_missing_module(stderr: str) -> str | None:
        match = re.search(r"No module named ['\"]([^'\"]+)['\"]", stderr)
        if match:
            return match.group(1)
        return None

    @staticmethod
    def _map_package(module_name: str) -> str | None:
        mapping = {
            "docx": "python-docx",
            "win32com": "pywin32",
        }
        return mapping.get(module_name)

    @staticmethod
    def _powershell_script_incomplete(script: str) -> bool:
        stripped = script.strip()
        if not stripped:
            return True
        if stripped.endswith("{") or stripped.endswith("`") or stripped.endswith("if") or stripped.endswith("try"):
            return True
        return False

    def _log_exec_result(self, result: dict[str, Any]) -> None:
        if not DEBUG:
            return
        stdout = (result.get("stdout") or "")[:1000]
        stderr = (result.get("stderr") or "")[:400]
        self._log_debug("SCRIPT_PATH", str(result.get("script_path")))
        self._log_debug("EXEC", str(result.get("exec_cmd")))
        if stdout:
            self._log_debug("STDOUT", stdout)
        if stderr:
            self._log_debug("STDERR_HEAD", stderr)
        self._log_debug("RC", f"{result.get('returncode')}")
        self._log_debug("DURATION_MS", f"{result.get('duration_ms')}")

    def run(self, user_input: str) -> str:
        self.logger.log_user_input(user_input, len(self.messages))
        self.logger.log("user_input", {"content": user_input})
        self.messages.append({"role": "user", "content": user_input})
        request_confirmed = False
        last_language: str | None = None
        last_risk: RiskLevel | None = None
        risk_rank = {
            RiskLevel.LOW: 0,
            RiskLevel.MEDIUM: 1,
            RiskLevel.HIGH: 2,
        }

        for attempt in range(MAX_RETRIES + 1):
            response = self.client.chat(self.messages, tools=[], tool_choice="none")
            raw_content = response.choices[0].message.content or ""
            self._log_debug("LLM_RAW", sanitize_assistant_text(raw_content)[:300])
            language, script = self._extract_script(raw_content)
            if not language or not script:
                if self._is_command_text(raw_content):
                    language = "powershell"
                    script = raw_content.strip()
                else:
                    return sanitize_assistant_text(raw_content)
            self._log_debug("SCRIPT_LANG", language)
            self._log_debug("SCRIPT_HASH", self._script_hash(script))

            if language == "python":
                try:
                    compile(script, "<agent>", "exec")
                except SyntaxError as exc:
                    self.messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"Скрипт синтаксически неверный: {exc}. "
                                "Верни исправленный полный python-код одним блоком."
                            ),
                        }
                    )
                    continue
            if language == "powershell" and self._powershell_script_incomplete(script):
                self.messages.append(
                    {
                        "role": "user",
                        "content": (
                            "Скрипт PowerShell выглядит неполным или пустым. "
                            "Верни исправленный полный powershell-код одним блоком."
                        ),
                    }
                )
                continue

            current_risk = assess_risk(language, script)
            need_confirm = True
            if request_confirmed and last_language == language and last_risk:
                if risk_rank[current_risk] <= risk_rank[last_risk]:
                    need_confirm = False
            self._log_debug("CONFIRM", "repeat" if need_confirm else "reuse")

            if need_confirm:
                approved = confirm_if_needed(language, script)
                self.logger.log_policy("SCRIPT", "script_execution", approved)
                if not approved:
                    return "Ок, отменено пользователем."
                request_confirmed = True
                last_language = language
                last_risk = current_risk

            result = self._run_script(language, script)
            self._log_exec_result(result)
            stdout = (result.get("stdout") or "")[:2000]
            stderr = (result.get("stderr") or "")[:2000]
            ok = bool(result.get("ok"))
            returncode = result.get("returncode")

            if language == "python" and not ok:
                missing_module = self._extract_missing_module(stderr)
                if missing_module:
                    package = self._map_package(missing_module)
                    if package:
                        self._log_debug("MISSING_DEP", f"{missing_module} -> {package}")
                        install = input(f"Не установлен пакет {package}. Установить? (y/n): ").strip().lower()
                        if install == "y":
                            install_result = run_pip_install(package, TIMEOUT_SEC)
                            self._log_exec_result(install_result)
                            if install_result.get("ok"):
                                self.messages.append({"role": "user", "content": user_input})
                                continue

            if ok:
                summary = "✅ Готово"
                if stdout or stderr:
                    return f"{summary}\nreturncode={returncode}\nstdout=\n{stdout}\nstderr=\n{stderr}"
                return f"{summary}\nreturncode={returncode}"

            if attempt < MAX_RETRIES:
                error_message = (
                    "Скрипт упал. Вот результат выполнения:\n"
                    f"returncode={returncode}\n"
                    f"stdout=\n{stdout}\n"
                    f"stderr=\n{stderr}\n"
                    "Исправь скрипт и верни ТОЛЬКО новый код-блок."
                )
                self.messages.append({"role": "user", "content": error_message})
                continue

            return (
                "❌ Ошибка выполнения\n"
                f"returncode={returncode}\nstdout=\n{stdout}\nstderr=\n{stderr}"
            )

        return ""
