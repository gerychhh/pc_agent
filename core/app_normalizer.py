from __future__ import annotations

import re

from .app_aliases import get_alias
from .config import FAST_MODEL
from .llm_client import LLMClient


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
    "дота": "dota 2",
    "доту": "dota 2",
    "dota": "dota 2",
    "блокнот": "notepad",
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


def normalize_alias_key(text: str) -> str:
    cleaned = text.lower()
    for token in _REMOVE_TOKENS:
        cleaned = cleaned.replace(token, " ")
    cleaned = " ".join(cleaned.split())
    parts = cleaned.split()
    normalized_parts = [_NUMBER_WORDS.get(part, part) for part in parts]
    return " ".join(normalized_parts).strip()


def normalize_app_query(text: str, use_llm: bool = True) -> str:
    key = normalize_alias_key(text)
    learned = get_alias(key)
    if learned:
        return learned
    if use_llm:
        suggestion = _suggest_app_name(text)
        if suggestion:
            return suggestion
    if key in _ALIASES:
        return _ALIASES[key]
    return _ALIASES.get(key, key)


def _suggest_app_name(text: str) -> str | None:
    if not text.strip():
        return None
    prompt = (
        "Ты помощник для запуска приложений через Windows Search. "
        "Верни только корректное название приложения или игры для поиска. "
        "Никаких пояснений, только одно короткое название. "
        "Если запрос не относится к запуску приложения, верни пустую строку."
    )
    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Запрос: {text}"},
    ]
    try:
        response = LLMClient().chat(messages, tools=[], model_name=FAST_MODEL, tool_choice="none")
    except Exception:
        return None
    content = response.choices[0].message.content or ""
    cleaned = _clean_llm_response(content)
    return cleaned or None


def _clean_llm_response(text: str) -> str:
    cleaned = text.strip().strip('"').strip("'")
    cleaned = re.sub(r"^[^A-Za-zА-Яа-я0-9]+", "", cleaned)
    cleaned = re.sub(r"[^A-Za-zА-Яа-я0-9]+$", "", cleaned)
    return cleaned.strip()
