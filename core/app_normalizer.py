from __future__ import annotations


_REMOVE_TOKENS = ("открой", "запусти", "включи", "пожалуйста", "просто")
_NUMBER_WORDS = {
    "ноль": "0",
    "один": "1",
    "одна": "1",
    "два": "2",
    "две": "2",
    "три": "3",
    "четыре": "4",
    "пять": "5",
    "шесть": "6",
    "семь": "7",
    "восемь": "8",
    "девять": "9",
    "десять": "10",
}
_ALIASES = {
    "дота 2": "dota 2",
    "дота два": "dota 2",
    "ворд": "word",
    "эксель": "excel",
    "дискорд": "discord",
    "телега": "telegram",
    "телеграм": "telegram",
    "хром": "chrome",
    "стим": "steam",
    "яндекс музыка": "yandex music",
    "фотошоп": "photoshop",
    "премьера": "premiere",
    "панель управления": "control panel",
}


def normalize_app_query(text: str) -> str:
    cleaned = text.lower()
    for token in _REMOVE_TOKENS:
        cleaned = cleaned.replace(token, " ")
    cleaned = " ".join(cleaned.split())
    if cleaned in _ALIASES:
        return _ALIASES[cleaned]
    parts = cleaned.split()
    normalized_parts = []
    for part in parts:
        normalized_parts.append(_NUMBER_WORDS.get(part, part))
    normalized = " ".join(normalized_parts).strip()
    return _ALIASES.get(normalized, normalized)
