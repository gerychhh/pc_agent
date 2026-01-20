from __future__ import annotations

import re


FORBIDDEN_PATHS = (
    r"c:\\windows",
    r"c:\\program files",
    r"c:\\program files (x86)",
    r"c:\\programdata",
    r"\\appdata\\",
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

    if re.search(r"\bwinreg\b|\bwin32service\b|\bwin32serviceutil\b", script, re.IGNORECASE):
        errors.append("Операции с реестром или службами запрещены.")

    if re.search(r"reg\s+add|reg\s+delete", script, re.IGNORECASE):
        errors.append("Изменение реестра запрещено.")

    if re.search(r"https?://\S+\.(exe|msi)", script, re.IGNORECASE):
        errors.append("Скачивание/запуск exe из интернета запрещено.")

    if _mentions_forbidden_paths(script):
        errors.append("Доступ к системным каталогам запрещен.")

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

    if re.search(r"Format-Volume|diskpart|bcdedit", script, re.IGNORECASE):
        errors.append("Операции с дисками/загрузчиком запрещены.")

    if re.search(r"\b(reg\s+add|reg\s+delete|Set-ItemProperty)\b", script, re.IGNORECASE):
        errors.append("Изменение реестра запрещено.")

    if re.search(r"\bsc\s+(stop|delete)\b|\bnet\s+stop\b", script, re.IGNORECASE):
        errors.append("Управление службами запрещено.")

    if re.search(r"\bshutdown\b|Restart-Computer", script, re.IGNORECASE):
        errors.append("Перезагрузка/выключение запрещены.")

    if re.search(r"\btakeown\b|\bicacls\b", script, re.IGNORECASE):
        errors.append("Изменение прав доступа запрещено.")

    if re.search(r"https?://\S+\.(exe|msi)", script, re.IGNORECASE):
        errors.append("Скачивание/запуск exe из интернета запрещено.")

    if _mentions_forbidden_paths(script):
        errors.append("Доступ к системным каталогам запрещен.")

    if re.search(r"Set-Location\s+['\"]Desktop['\"]", script, re.IGNORECASE):
        errors.append(
            "Нельзя использовать Set-Location \"Desktop\". Используй GetFolderPath('Desktop') и Join-Path."
        )

    return errors


def _mentions_forbidden_paths(script: str) -> bool:
    lowered = script.lower()
    return any(token in lowered for token in FORBIDDEN_PATHS)
