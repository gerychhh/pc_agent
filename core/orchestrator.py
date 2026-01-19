from __future__ import annotations

import hashlib
import re
from typing import Any

from .config import DEBUG, MAX_RETRIES, TIMEOUT_SEC
from .executor import run_pip_install, run_powershell, run_python
from .llm_client import LLMClient
from .logger import SessionLogger
from .policy import RiskLevel, assess_risk, confirm_if_needed


SYSTEM_PROMPT = (
    "Ты локальный ассистент управления ПК. "
    "Если запрос не требует выполнения действий, отвечай обычным текстом без кода. "
    "Если нужно выполнить действие, отвечай ТОЛЬКО одним код-блоком (python или powershell). "
    "Не пиши объяснений вне блока, когда приводишь код. "
    "Возвращай ТОЛЬКО один полный code block. "
    "Код должен быть завершённым и исполняемым. Никаких обрывков вроде 'except ImportError:' без тела. "
    "Если нужна операция с файлами — используй абсолютные пути и сначала получи Desktop путь "
    "через Python: os.path.join(os.path.expanduser('~'),'Desktop'). "
    "Если нужно создать DOCX — используй python-docx (from docx import Document). "
    "Не используй win32com, если явно не сказано. "
    "Любые разрушительные действия запрещены: удаление системных файлов, форматирование, "
    "отключение защиты, загрузка подозрительных файлов. "
    "Если не уверен — спроси пользователя уточнить путь."
)


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
