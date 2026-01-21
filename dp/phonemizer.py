from __future__ import annotations


class Phonemizer:
    """Minimal fallback for DeepPhonemizer's interface used by openWakeWord."""

    def __init__(self, fallback_phonemes: str = "AH") -> None:
        self._fallback_phonemes = fallback_phonemes

    @classmethod
    def from_checkpoint(cls, _path: str) -> "Phonemizer":
        return cls()

    def __call__(self, _text: str, lang: str | None = None) -> str:
        return self._fallback_phonemes
