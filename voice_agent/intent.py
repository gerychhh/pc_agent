from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class Intent:
    name: str
    slots: dict[str, str]
    confidence: float


class IntentRecognizer:
    def __init__(self, synonyms: dict[str, str]) -> None:
        self.synonyms = synonyms
        self._wake_words = ("агент", "assistant")
        self._patterns = [
            ("open_app", re.compile(r"^(открой|запусти|включи)\s+(?P<app>.+)$")),
            ("close_app", re.compile(r"^(закрой|выключи)\s+(?P<app>.+)$")),
            ("type_text", re.compile(r"^(набери|введи)\s+(?P<text>.+)$")),
            ("search_web", re.compile(r"^(найди|поиск)\s+(?P<query>.+)$")),
            ("paste_text", re.compile(r"^(вставь|вклей)\s+(?P<text>.+)$")),
        ]

    def normalize(self, text: str) -> str:
        normalized = re.sub(r"[^\w\s]+", " ", text.strip().lower())
        normalized = re.sub(r"\s+", " ", normalized).strip()
        for wake in self._wake_words:
            if normalized.startswith(wake + " "):
                normalized = normalized[len(wake) + 1 :]
        for key, value in self.synonyms.items():
            normalized = normalized.replace(key, value)
        return normalized

    def recognize(self, text: str) -> Intent | None:
        normalized = self.normalize(text)
        for name, pattern in self._patterns:
            match = pattern.match(normalized)
            if match:
                return Intent(name=name, slots=match.groupdict(), confidence=0.85)
        return None
