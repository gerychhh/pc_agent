from __future__ import annotations


def route_task(user_text: str) -> dict[str, str | None]:
    lowered = user_text.lower()
    multi_step_markers = (
        " и ",
        " затем ",
        " потом ",
    )
    ui_keywords = (
        "напиши",
        "введи",
        "нажми",
        "нажать",
        "клик",
        "кликни",
        "перемотай",
        "пауза",
        "громче",
        "тише",
        "поставь на паузу",
    )
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
    if any(keyword in lowered for keyword in ui_keywords):
        complexity = "complex" if any(marker in lowered for marker in multi_step_markers) else "simple"
        return {"complexity": complexity, "force_lang": "python", "reason": "ui_keywords"}
    if any(keyword in lowered for keyword in complex_keywords):
        return {"complexity": "complex", "force_lang": "python", "reason": "complex_keywords"}
    if any(keyword in lowered for keyword in simple_keywords):
        complexity = "complex" if any(marker in lowered for marker in multi_step_markers) else "simple"
        return {"complexity": complexity, "force_lang": None, "reason": "simple_keywords"}
    return {"complexity": "simple", "force_lang": None, "reason": "default"}
