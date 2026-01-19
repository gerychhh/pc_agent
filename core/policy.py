from __future__ import annotations

from enum import Enum


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"


_POWERSHELL_HIGH_RISK = {
    "remove-item",
    "del ",
    "rd ",
    "format",
    "diskpart",
    "bcdedit",
    "shutdown",
    "reg delete",
}

_PYTHON_MED_RISK = {
    "os.remove",
    "os.rmdir",
    "shutil.rmtree",
    "subprocess",
}


def _summarize_script(script_text: str, max_lines: int = 10) -> str:
    lines = script_text.strip().splitlines()
    return "\n".join(lines[:max_lines])


def assess_risk(language: str, script_text: str) -> RiskLevel:
    lowered = script_text.lower()
    if language == "powershell":
        if any(token in lowered for token in _POWERSHELL_HIGH_RISK):
            return RiskLevel.HIGH
        return RiskLevel.MEDIUM
    if language == "python":
        if any(token in lowered for token in _PYTHON_MED_RISK):
            return RiskLevel.MEDIUM
        return RiskLevel.LOW
    return RiskLevel.LOW


def confirm_if_needed(language: str, script_text: str) -> bool:
    level = assess_risk(language, script_text)
    summary = _summarize_script(script_text)
    print("\n⚠️  Action confirmation required")
    print(f"Language: {language}")
    print("Script preview:")
    print(summary if summary else "(empty script)")
    if level == RiskLevel.HIGH:
        print("Warning: high risk script detected.")
    elif level == RiskLevel.MEDIUM:
        print("Warning: potentially risky operations detected.")
    print("Confirm? (y/n): ", end="", flush=True)
    response = input().strip().lower()
    return response == "y"
