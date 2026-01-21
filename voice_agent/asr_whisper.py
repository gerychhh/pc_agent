from __future__ import annotations

import time
from dataclasses import dataclass
import logging

import numpy as np
from faster_whisper import WhisperModel

from .bus import Event, EventBus


@dataclass(frozen=True)
class AsrConfig:
    model: str
    device: str
    compute_type: str
    beam_size: int
    language: str
    max_utterance_s: int
    partial_interval_ms: int
    partial_min_delta: int


class FasterWhisperASR:
    def __init__(self, config: AsrConfig, bus: EventBus) -> None:
        self.config = config
        self.bus = bus
        self.logger = logging.getLogger("voice_agent")
        self._fallback_attempted = False
        self._load_model(config.device, config.compute_type)
        self._buffer: list[np.ndarray] = []
        self._last_partial: str = ""
        self._last_partial_emit = 0.0
        self._active = False

    def _load_model(self, device: str, compute_type: str) -> None:
        try:
            self.model = WhisperModel(
                self.config.model,
                device=device,
                compute_type=compute_type,
            )
            self.logger.info("ASR model loaded on %s (%s).", device, compute_type)
        except (RuntimeError, OSError) as exc:
            message = str(exc).lower()
            if device == "cuda" and ("cublas" in message or "cuda" in message):
                self.logger.warning(
                    "ASR GPU init failed (%s). Falling back to CPU.",
                    exc,
                )
                self._fallback_attempted = True
                self.model = WhisperModel(
                    self.config.model,
                    device="cpu",
                    compute_type="int8",
                )
                self.logger.info("ASR model loaded on cpu (int8).")
            else:
                raise

    def reset(self) -> None:
        self._buffer = []
        self._last_partial = ""
        self._last_partial_emit = 0.0
        self._active = False

    def speech_start(self) -> None:
        self.reset()
        self._active = True

    def speech_end(self, ts: float) -> None:
        if not self._active:
            return
        text = self._transcribe(self._buffer)
        if text:
            self.bus.publish(Event("asr.final", {"text": text, "ts": ts}))
        self.reset()

    def accept_audio(self, chunk: np.ndarray, ts: float) -> None:
        if not self._active:
            return
        self._buffer.append(chunk.copy())
        if self._should_emit_partial(ts):
            text = self._transcribe(self._buffer)
            if self._is_significant_partial(text):
                self._last_partial = text
                self._last_partial_emit = ts
                self.bus.publish(Event("asr.partial", {"text": text, "ts": ts, "stability": 0.5}))

    def _should_emit_partial(self, ts: float) -> bool:
        return (ts - self._last_partial_emit) * 1000.0 >= self.config.partial_interval_ms

    def _is_significant_partial(self, text: str) -> bool:
        if not text:
            return False
        if text == self._last_partial:
            return False
        return abs(len(text) - len(self._last_partial)) >= self.config.partial_min_delta

    def _run_transcribe(self, audio: np.ndarray) -> str:
        segments, _info = self.model.transcribe(
            audio,
            language=self.config.language,
            beam_size=self.config.beam_size,
            vad_filter=False,
        )
        return "".join(segment.text for segment in segments).strip()

    def _transcribe(self, chunks: list[np.ndarray]) -> str:
        if not chunks:
            return ""
        audio = np.concatenate(chunks, axis=0).astype(np.float32) / 32768.0
        if audio.ndim > 1:
            audio = audio[:, 0]
        try:
            return self._run_transcribe(audio)
        except RuntimeError as exc:
            message = str(exc).lower()
            if not self._fallback_attempted and ("cublas" in message or "cuda" in message):
                self.logger.warning("ASR transcribe failed on GPU: %s", exc)
                self._fallback_attempted = True
                self._load_model("cpu", "int8")
                try:
                    return self._run_transcribe(audio)
                except RuntimeError as retry_exc:
                    self.logger.error("ASR transcribe failed after CPU fallback: %s", retry_exc)
                    return ""
            self.logger.error("ASR transcribe failed: %s", exc)
            return ""
