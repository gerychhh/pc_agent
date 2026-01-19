from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _result(ok: bool, **kwargs: Any) -> str:
    payload = {"ok": ok, **kwargs}
    if not ok and "error" not in payload:
        payload["error"] = "unknown_error"
    return json.dumps(payload, ensure_ascii=False)


def read_file(path: str, max_chars: int = 20000) -> str:
    try:
        target = Path(path)
        content = target.read_text(encoding="utf-8")
        if len(content) > max_chars:
            content = content[:max_chars]
        return _result(True, content=content)
    except Exception as exc:
        return _result(False, error=str(exc))


def write_file(path: str, content: str) -> str:
    try:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
        return _result(True, path=str(target))
    except Exception as exc:
        return _result(False, error=str(exc))
