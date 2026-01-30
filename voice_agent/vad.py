from __future__ import annotations

import logging
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
    min_rms: float = 0.01
    noise_floor_alpha: float = 0.05
    noise_ratio: float = 1.5
    score_emit_ms: int = 120


class SileroVAD:
    """Streaming VAD based on snakers4/silero-vad.

    Events:
      - vad.speech_start: {"ts": float}
      - vad.speech_end:   {"ts": float}
      - vad.score: {"ts": float, "prob": float, "rms": float, "noise_gate": float, "speaking": bool}
    """

    def __init__(self, config: VadConfig, bus: EventBus) -> None:
        self.config = config
        self.bus = bus
        self.logger = logging.getLogger("voice_agent.vad")

        self.device = torch.device(config.device)
        model, _utils = torch.hub.load(repo_or_dir="snakers4/silero-vad", model="silero_vad")
        model.to(self.device)
        model.eval()
        self._model = model

        self._state = None
        self._speaking = False
        self._speech_started_at: float | None = None
        self._last_voice_at: float | None = None
        self._audio_buffer = np.zeros(0, dtype=np.float32)

        # Silero expects 512 samples @ 16kHz (≈32ms) by default
        self._min_samples = int(self.config.sample_rate / 31.25)

        self._noise_floor_rms: float | None = None
        self._last_score_emit_ts = 0.0

        self.logger.info(
            "VAD loaded device=%s threshold=%.2f min_speech_ms=%d end_silence_ms=%d",
            self.config.device,
            self.config.threshold,
            self.config.min_speech_ms,
            self.config.end_silence_ms,
        )

    def reset(self) -> None:
        self._state = None
        self._speaking = False
        self._speech_started_at = None
        self._last_voice_at = None
        self._audio_buffer = np.zeros(0, dtype=np.float32)
        self._noise_floor_rms = None
        self._last_score_emit_ts = 0.0

    def process_chunk(self, chunk: np.ndarray, ts: float) -> None:
        if chunk.size == 0:
            return

        audio = chunk.astype(np.float32) / 32768.0
        if audio.ndim > 1:
            audio = audio[:, 0]

        self._audio_buffer = np.concatenate([self._audio_buffer, audio])

        while self._audio_buffer.size >= self._min_samples:
            frame = self._audio_buffer[: self._min_samples]
            self._audio_buffer = self._audio_buffer[self._min_samples :]

            rms = float(np.sqrt(np.mean(frame * frame))) if frame.size else 0.0

            # Estimate noise floor when NOT speaking
            if not self._speaking and rms > 0:
                if self._noise_floor_rms is None:
                    self._noise_floor_rms = rms
                else:
                    self._noise_floor_rms = (
                        (1.0 - self.config.noise_floor_alpha) * self._noise_floor_rms
                        + self.config.noise_floor_alpha * rms
                    )

            audio_tensor = torch.from_numpy(frame).to(self.device)
            with torch.no_grad():
                prob = float(self._model(audio_tensor, self.config.sample_rate).item())

            noise_gate = max(float(self.config.min_rms), float((self._noise_floor_rms or 0.0) * float(self.config.noise_ratio)))

            # periodic score event for UI/logs
            if (ts - self._last_score_emit_ts) * 1000.0 >= int(self.config.score_emit_ms):
                self.bus.publish(Event("vad.score", {"ts": ts, "prob": prob, "rms": rms, "noise_gate": noise_gate, "speaking": self._speaking}))
                self._last_score_emit_ts = ts

            if prob >= float(self.config.threshold) and rms >= noise_gate:
                self._last_voice_at = ts
                if not self._speaking:
                    self._speech_started_at = ts
                    self._speaking = True
                    self.logger.info("VAD speech_start (prob=%.2f rms=%.4f gate=%.4f)", prob, rms, noise_gate)
                    self.bus.publish(Event("vad.speech_start", {"ts": ts}))
            else:
                if self._speaking and self._last_voice_at is not None:
                    silence_ms = (ts - self._last_voice_at) * 1000.0
                    speech_ms = (ts - (self._speech_started_at or ts)) * 1000.0
                    if silence_ms >= int(self.config.end_silence_ms) and speech_ms >= int(self.config.min_speech_ms):
                        self._speaking = False
                        self._speech_started_at = None
                        self.logger.info("VAD speech_end (silence_ms=%d speech_ms=%d)", int(silence_ms), int(speech_ms))
                        self.bus.publish(Event("vad.speech_end", {"ts": ts}))
