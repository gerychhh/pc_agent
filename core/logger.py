from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from .config import LOG_DIR


class LogLevel(str, Enum):
    INFO = "INFO"
    DEBUG = "DEBUG"
    WARN = "WARN"
    ERROR = "ERROR"


@dataclass
class SessionLogger:
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    file_path: Path = field(init=False)
    debug_enabled: bool = field(default_factory=lambda: os.getenv("DEBUG") == "1")

    def __post_init__(self) -> None:
        self.file_path = LOG_DIR / f"session_{self.session_id}.jsonl"

    def log(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            "ts": datetime.now().isoformat(),
            "event": event_type,
            "payload": payload,
        }
        with self.file_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    def _emit(self, level: LogLevel, message: str) -> None:
        if not self.debug_enabled:
            return
        print(f"[{level.value}] {message}")

    @staticmethod
    def _truncate(value: str, limit: int) -> str:
        if len(value) <= limit:
            return value
        return value[:limit] + "..."

    def log_user_input(self, content: str, context_len: int) -> None:
        self._emit(LogLevel.DEBUG, f"[USER] {content}")
        self._emit(LogLevel.DEBUG, f"[CTX] {context_len}")

    def log_llm_response(self, content: str, tool_calls: list[dict[str, Any]]) -> None:
        snippet = self._truncate(content or "", 400)
        if snippet:
            self._emit(LogLevel.DEBUG, f"[LLM] {snippet}")
        self._emit(LogLevel.DEBUG, f"[LLM] tool_calls {json.dumps(tool_calls, ensure_ascii=False)}")

    def log_policy(self, level: str, reason: str, approved: bool) -> None:
        decision = "y" if approved else "n"
        self._emit(LogLevel.DEBUG, f"[POLICY] level={level} reason={reason} decision={decision}")

    def log_tool_run(
        self,
        name: str,
        args: dict[str, Any],
        exec_cmd: str | None,
        duration_ms: int | None,
        stdout: str | None,
        stderr: str | None,
        ok: bool | None,
        verified: bool | None,
    ) -> None:
        self._emit(LogLevel.DEBUG, f"[TOOL] {name}")
        self._emit(LogLevel.DEBUG, f"[ARGS] {json.dumps(args, ensure_ascii=False, indent=2)}")
        if exec_cmd:
            self._emit(LogLevel.DEBUG, f"[EXEC] {exec_cmd}")
        if duration_ms is not None:
            self._emit(LogLevel.DEBUG, f"[TIME] {duration_ms}")
        if stdout:
            self._emit(LogLevel.DEBUG, f"[STDOUT] {self._truncate(stdout, 500)}")
        if stderr:
            self._emit(LogLevel.DEBUG, f"[STDERR] {self._truncate(stderr, 500)}")
        if ok is not None or verified is not None:
            self._emit(LogLevel.DEBUG, f"[RESULT] ok={ok} verified={verified}")

    def log_tool_summary(
        self,
        name: str,
        args: dict[str, Any],
        ok: bool | None,
        verified: bool | None,
        stdout: str | None,
        stderr: str | None,
    ) -> None:
        self._emit(LogLevel.DEBUG, f"[TOOL] {name} args={json.dumps(args, ensure_ascii=False)}")
        self._emit(LogLevel.DEBUG, f"[TOOL] ok={ok} verified={verified}")
        if stdout:
            self._emit(LogLevel.DEBUG, f"[TOOL] stdout={self._truncate(stdout, 500)}")
        if stderr:
            self._emit(LogLevel.DEBUG, f"[TOOL] stderr={self._truncate(stderr, 500)}")

    def log_final(self, content: str) -> None:
        self._emit(LogLevel.DEBUG, f"[FINAL] {content}")

    def warn(self, message: str) -> None:
        self._emit(LogLevel.WARN, message)

    def error(self, message: str) -> None:
        self._emit(LogLevel.ERROR, message)
