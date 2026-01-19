from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from .config import PROJECT_ROOT


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


PROCESS_ACTIONS = {
    "open_app",
    "open_url",
    "run_cmd",
    "run_powershell",
    "run_python_script",
}
FILE_WRITE_ACTIONS = {
    "write_text_file_lines",
    "create_docx",
}


def risk_level(tool_name: str, args: dict[str, Any]) -> RiskLevel:
    if tool_name in PROCESS_ACTIONS or tool_name in FILE_WRITE_ACTIONS:
        return RiskLevel.MEDIUM

    if tool_name == "write_file":
        path = str(args.get("path", ""))
        lower_path = path.lower()
        if "windows" in lower_path or "program files" in lower_path:
            return RiskLevel.HIGH
        try:
            target = Path(path).resolve()
            if PROJECT_ROOT not in target.parents and target != PROJECT_ROOT:
                return RiskLevel.MEDIUM
        except OSError:
            return RiskLevel.MEDIUM

    return RiskLevel.LOW


def risk_reason(tool_name: str, args: dict[str, Any], level: RiskLevel) -> str:
    if tool_name in PROCESS_ACTIONS:
        return "process_action"
    if tool_name in FILE_WRITE_ACTIONS:
        return "file_write"
    if tool_name == "write_file":
        path = str(args.get("path", ""))
        lower_path = path.lower()
        if "windows" in lower_path or "program files" in lower_path:
            return "system_path_write"
        return "write_file_outside_project"
    if level == RiskLevel.LOW:
        return "low_risk"
    return "unclassified"


def confirm_action(tool_name: str, args: dict[str, Any], level: RiskLevel) -> bool:
    if level == RiskLevel.LOW:
        return True

    pretty_args = json.dumps(args, ensure_ascii=False)
    print("\n⚠️  Action confirmation required")
    print(f"Action requested: {tool_name} {pretty_args}")
    if level == RiskLevel.HIGH:
        print("Warning: this action is high risk and may impact system files.")
    print("Confirm? (y/n): ", end="", flush=True)
    response = input().strip().lower()
    return response == "y"
