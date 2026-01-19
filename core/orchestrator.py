from __future__ import annotations

import re
from typing import Any

from .config import DEBUG, MAX_RETRIES, TIMEOUT_SEC
from .executor import run_powershell, run_python
from .llm_client import LLMClient
from .logger import SessionLogger
from .policy import confirm_if_needed


SYSTEM_PROMPT = (
    "Ты локальный ассистент управления ПК. "
    "Всегда отвечай ТОЛЬКО одним код-блоком (python или powershell). "
    "Не пиши объяснений вне блока. "
    "Если нужна операция с файлами — используй абсолютные пути и сначала получи Desktop путь "
    "через Python: os.path.join(os.path.expanduser('~'),'Desktop'). "
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

    def _log_exec_result(self, result: dict[str, Any]) -> None:
        if not DEBUG:
            return
        stdout = (result.get("stdout") or "")[:1000]
        stderr = (result.get("stderr") or "")[:1000]
        self._log_debug("SCRIPT_PATH", str(result.get("script_path")))
        self._log_debug("EXEC", str(result.get("exec_cmd")))
        if stdout:
            self._log_debug("STDOUT", stdout)
        if stderr:
            self._log_debug("STDERR", stderr)
        self._log_debug("RC", f"{result.get('returncode')} duration_ms={result.get('duration_ms')}")

    def run(self, user_input: str) -> str:
        self.logger.log_user_input(user_input, len(self.messages))
        self.logger.log("user_input", {"content": user_input})
        self.messages.append({"role": "user", "content": user_input})

        for attempt in range(MAX_RETRIES + 1):
            response = self.client.chat(self.messages, tools=[], tool_choice="none")
            raw_content = response.choices[0].message.content or ""
            self._log_debug("LLM_RAW", sanitize_assistant_text(raw_content)[:300])
            language, script = self._extract_script(raw_content)
            if not language or not script:
                return "LLM не дал скрипт"
            self._log_debug("SCRIPT_LANG", language)

            approved = confirm_if_needed(language, script)
            self.logger.log_policy("SCRIPT", "script_execution", approved)
            if not approved:
                return "Ок, отменено пользователем."

            result = self._run_script(language, script)
            self._log_exec_result(result)
            stdout = (result.get("stdout") or "")[:2000]
            stderr = (result.get("stderr") or "")[:2000]
            ok = bool(result.get("ok"))
            returncode = result.get("returncode")

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
