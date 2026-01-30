from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bus import Event, EventBus


@dataclass(frozen=True)
class AsrVoskConfig:
    model_path: Path
    sample_rate: int = 16000
    max_utterance_s: int = 10
    partial_interval_ms: int = 220
    partial_min_delta: int = 3
    min_buffer_s: float = 0.4


class VoskASR:
    """Streaming ASR backend using Vosk.

    Interface matches FasterWhisperASR enough for VoiceAgentRuntime.
    """

    def __init__(self, config: AsrVoskConfig, bus: EventBus) -> None:
        self.config = config
        self.bus = bus
        self.logger = logging.getLogger("voice_agent.asr_vosk")

        try:
            import vosk  # type: ignore
            vosk.SetLogLevel(-1)
        except Exception as exc:
            raise ImportError("vosk is not installed. pip install vosk") from exc

        if not self.config.model_path.exists():
            raise FileNotFoundError(f"Vosk model not found: {self.config.model_path}")

        self._vosk = vosk
        self._model = vosk.Model(str(self.config.model_path))
        self.logger.info("ASR(Vosk) model loaded: %s", self.config.model_path)

        self._rec = None
        self._active = False
        self._speech_start_ts: float | None = None
        self._last_partial_emit = 0.0
        self._last_partial = ""
        self._samples_total = 0

    def reset(self) -> None:
        self._rec = None
        self._active = False
        self._speech_start_ts = None
        self._last_partial_emit = 0.0
        self._last_partial = ""
        self._samples_total = 0

    def speech_start(self) -> None:
        self.reset()
        self._active = True
        self._speech_start_ts = time.monotonic()
        self._rec = self._vosk.KaldiRecognizer(self._model, self.config.sample_rate)
        self._rec.SetWords(False)

    def accept_audio(self, chunk: np.ndarray, ts: float) -> None:
        if not self._active or not self._rec:
            return

        if chunk.ndim > 1:
            chunk = chunk[:, 0]
        if chunk.dtype != np.int16:
            chunk = chunk.astype(np.int16, copy=False)

        self._samples_total += int(chunk.shape[0])

        # max utterance cutoff
        if self._speech_start_ts and (ts - self._speech_start_ts) >= float(self.config.max_utterance_s):
            self.logger.info("ASR(Vosk) max utterance reached, forcing final.")
            self._finalize(ts)
            return

        self._rec.AcceptWaveform(chunk.tobytes())

        if (ts - self._last_partial_emit) * 1000.0 >= int(self.config.partial_interval_ms):
            self._last_partial_emit = ts
            try:
                partial = json.loads(self._rec.PartialResult() or "{}").get("partial", "").strip()
            except Exception:
                partial = ""
            if self._is_significant_partial(partial):
                self._last_partial = partial
                self.bus.publish(Event("asr.partial", {"text": partial, "ts": ts, "stability": 0.3}))

    def speech_end(self, ts: float) -> None:
        if not self._active or not self._rec:
            return
        self._finalize(ts)

    def _buffer_duration_s(self) -> float:
        return float(self._samples_total) / float(self.config.sample_rate)

    def _is_significant_partial(self, text: str) -> bool:
        if not text:
            return False
        if text == self._last_partial:
            return False
        return abs(len(text) - len(self._last_partial)) >= int(self.config.partial_min_delta)

    def _finalize(self, ts: float) -> None:
        if self._buffer_duration_s() < float(self.config.min_buffer_s):
            self.logger.info("ASR(Vosk) buffer too short (%.2fs), dropping.", self._buffer_duration_s())
            self.reset()
            return

        try:
            res = json.loads(self._rec.FinalResult() or "{}")
            text = (res.get("text") or "").strip()
        except Exception:
            text = ""

        if not text and self._last_partial:
            text = self._last_partial

        if text:
            self.bus.publish(Event("asr.final", {"text": text, "ts": ts}))
        self.reset()
