from __future__ import annotations

import re


def validate_python(script: str) -> list[str]:
    errors: list[str] = []
    lines = script.splitlines()
    if any(";" in line for line in lines):
        errors.append("Запрещены однострочники с ';' в Python.")
    if re.search(r"open\([^\n]*\.(docx|xlsx|pdf)[^\n]*['\"]w", script, re.IGNORECASE):
        errors.append("Нельзя создавать docx/xlsx/pdf через open(...,'w').")
    if "Path(" in script or "Path." in script:
        if "from pathlib import Path" not in script:
            errors.append("Если используется Path, нужен импорт: from pathlib import Path.")
    if re.search(r"\.docx", script, re.IGNORECASE):
        if "from docx import Document" not in script:
            errors.append("Для .docx нужен python-docx: from docx import Document.")
        if ".save(" not in script:
            errors.append("Для .docx нужен вызов doc.save(...).")
    return errors


def validate_powershell(script: str) -> list[str]:
    errors: list[str] = []
    for match in re.finditer(r"Start-Process\s+['\"]([^'\"]+)['\"]", script, re.IGNORECASE):
        url = match.group(1)
        if url.startswith("http") and not url.startswith(("http://", "https://")):
            errors.append("Start-Process должен использовать http/https URL.")
    if re.search(r"Stop-Process\s+-Name\s+notepad\.exe", script, re.IGNORECASE):
        errors.append("Stop-Process должен использовать имя 'notepad' без .exe.")
    if re.search(r"Remove-Item\s+.*-Recurse", script, re.IGNORECASE):
        errors.append("Remove-Item -Recurse запрещен.")
    if re.search(r"Format-Volume|Clear-Disk", script, re.IGNORECASE):
        errors.append("Команды форматирования диска запрещены.")
    if re.search(r"Set-Location\s+['\"]Desktop['\"]", script, re.IGNORECASE):
        errors.append(
            "Нельзя использовать Set-Location \"Desktop\". Используй GetFolderPath('Desktop') и Join-Path."
        )
    return errors
