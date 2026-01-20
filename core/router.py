from __future__ import annotations


def route_task(user_text: str) -> dict[str, str | None]:
    lowered = user_text.lower()
    complex_keywords = (
        "оформи",
        "по правилам",
        "гост",
        "стиль",
        "шрифт",
        "таблица",
        "сделай красиво",
        "документ",
        "docx",
        "xlsx",
        "pdf",
        "много шагов",
    )
    simple_keywords = (
        "открой",
        "закрой",
        "найди",
        "включи",
        "запусти",
        "пауза",
        "перемотай",
        "громче",
        "тише",
    )
    if any(keyword in lowered for keyword in complex_keywords):
        return {"complexity": "complex", "force_lang": "python", "reason": "complex_keywords"}
    if any(keyword in lowered for keyword in simple_keywords):
        return {"complexity": "simple", "force_lang": None, "reason": "simple_keywords"}
    return {"complexity": "simple", "force_lang": None, "reason": "default"}
