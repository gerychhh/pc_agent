from __future__ import annotations

import re


FORBIDDEN_PATHS = ()

SAFE_PATH_TOKENS = ()

WRITE_COMMANDS_PS = (
    "set-content",
    "add-content",
    "out-file",
    "new-item",
    "copy-item",
    "move-item",
)

WRITE_HINTS_PY = (
    "open(",
    "write_text(",
    "write_bytes(",
    "to_csv(",
    "to_json(",
)


def validate_python(script: str) -> list[str]:
    errors: list[str] = []
    lines = script.splitlines()

    if any(";" in line for line in lines):
        errors.append("Запрещены однострочники с ';' в Python.")

    if re.search(r"open\([^\n]*\.(docx|xlsx|pdf|pptx)[^\n]*['\"]w", script, re.IGNORECASE):
        errors.append("Нельзя создавать docx/xlsx/pptx/pdf через open(...,'w').")

    if "Path(" in script or "Path." in script:
        if "from pathlib import Path" not in script:
            errors.append("Если используется Path, нужен импорт: from pathlib import Path.")

    if re.search(r"\.docx", script, re.IGNORECASE):
        if "from docx import Document" not in script:
            errors.append("Для .docx нужен python-docx: from docx import Document.")
        if ".save(" not in script:
            errors.append("Для .docx нужен вызов doc.save(...).")

    if re.search(r"\b(os\.remove|os\.rmdir|shutil\.rmtree|Path\.unlink)\b", script):
        errors.append("Опасное удаление файлов/папок запрещено.")

    return errors


def validate_powershell(script: str) -> list[str]:
    errors: list[str] = []

    if re.search(
        r"\b(New-Item|Add-Content|Set-Content|Out-File|Get-Content)\b[^\n]*\.(docx|xlsx|pptx)\b",
        script,
        re.IGNORECASE,
    ):
        errors.append(
            "Нельзя создавать/править .docx/.xlsx/.pptx через текстовые PowerShell команды "
            "(New-Item/Add-Content/Set-Content/Out-File/Get-Content)."
        )

    if re.search(r"\b(Remove-Item|del\b|erase\b|rmdir\b|rd\b)\b", script, re.IGNORECASE):
        errors.append("Опасное удаление файлов/папок запрещено.")

    return errors


def _mentions_forbidden_paths(script: str) -> bool:
    lowered = script.lower()
    return any(token in lowered for token in FORBIDDEN_PATHS)


def _mentions_safe_paths(script: str) -> bool:
    lowered = script.lower()
    return any(token in lowered for token in SAFE_PATH_TOKENS)


def _uses_write_ops_ps(script: str) -> bool:
    lowered = script.lower()
    return any(token in lowered for token in WRITE_COMMANDS_PS)


def _uses_write_ops_py(script: str) -> bool:
    lowered = script.lower()
    return any(token in lowered for token in WRITE_HINTS_PY)
