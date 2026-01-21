from __future__ import annotations

from dataclasses import dataclass

from .bus import Event, EventBus


@dataclass(frozen=True)
class TtsConfig:
    enabled: bool
    voice: str
    engine: str


class TtsEngine:
    def __init__(self, config: TtsConfig, bus: EventBus) -> None:
        self.config = config
        self.bus = bus
        self._speaking = False

    def speak(self, text: str) -> None:
        if not self.config.enabled:
            return
        self._speaking = True
        self.bus.publish(Event("tts.start", {"text": text}))
        self.bus.publish(Event("tts.stop", {}))
        self._speaking = False

    def stop(self) -> None:
        if self._speaking:
            self._speaking = False
            self.bus.publish(Event("tts.stop", {}))
