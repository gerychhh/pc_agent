from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .bus import Event, EventBus


@dataclass(frozen=True)
class VadConfig:
    device: str = "cpu"
    threshold: float = 0.5
    min_speech_ms: int = 200
    end_silence_ms: int = 500
    sample_rate: int = 16000


class SileroVAD:
    def __init__(self, config: VadConfig, bus: EventBus) -> None:
        self.config = config
        self.bus = bus
        self.device = torch.device(config.device)
        model, utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
        model.to(self.device)
        model.eval()
        self._model = model
        self._state = None
        self._speaking = False
        self._speech_started_at: float | None = None
        self._last_voice_at: float | None = None

    def process_chunk(self, chunk: np.ndarray, ts: float) -> None:
        if chunk.size == 0:
            return
        audio = chunk.astype(np.float32) / 32768.0
        if audio.ndim > 1:
            audio = audio[:, 0]
        audio_tensor = torch.from_numpy(audio).to(self.device)
        with torch.no_grad():
            prob = self._model(audio_tensor, self.config.sample_rate).item()
        if prob >= self.config.threshold:
            self._last_voice_at = ts
            if not self._speaking:
                self._speech_started_at = ts
                self._speaking = True
                self.bus.publish(Event("vad.speech_start", {"ts": ts}))
        else:
            if self._speaking and self._last_voice_at is not None:
                silence_ms = (ts - self._last_voice_at) * 1000.0
                speech_ms = (ts - (self._speech_started_at or ts)) * 1000.0
                if silence_ms >= self.config.end_silence_ms and speech_ms >= self.config.min_speech_ms:
                    self._speaking = False
                    self._speech_started_at = None
                    self.bus.publish(Event("vad.speech_end", {"ts": ts}))
