from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import LOG_DIR


@dataclass
class SessionLogger:
    session_id: str = field(default_factory=lambda: datetime.now().strftime("%Y%m%d_%H%M%S"))
    file_path: Path = field(init=False)

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
