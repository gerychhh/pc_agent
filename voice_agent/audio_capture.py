from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import sounddevice as sd

from .bus import Event, EventBus


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int
    channels: int
    chunk_ms: int
    device: int | None = None


class AudioCapture:
    """Low-level microphone capture.

    Publishes:
      - audio.chunk: {"data": np.int16[...,1], "ts": float}
      - audio.level: {"rms": float(0..1), "peak": float(0..1), "ts": float}
    """

    def __init__(self, config: AudioConfig, bus: EventBus) -> None:
        self.config = config
        self.bus = bus
        self.logger = logging.getLogger("voice_agent.audio")
        self._stream: sd.InputStream | None = None
        self._last_level_emit = 0.0
        self._muted = False

    def set_muted(self, muted: bool) -> None:
        self._muted = muted

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        if status:
            # Do not crash the stream: some drivers report non-fatal flags.
            self.logger.warning("Audio stream status: %s", status)

        ts = time.monotonic()

        if not self._muted:
            self.bus.publish(Event("audio.chunk", {"data": indata.copy(), "ts": ts}))

        # Emit level frequently enough for UI, but not for every chunk.
        if ts - self._last_level_emit >= 0.12:
            if indata.size:
                x = indata.astype(np.float32)
                if x.ndim > 1:
                    x = x[:, 0]
                x /= 32768.0
                rms = float(np.sqrt(np.mean(x * x)))
                peak = float(np.max(np.abs(x)))
            else:
                rms = 0.0
                peak = 0.0
            self.bus.publish(Event("audio.level", {"rms": rms, "peak": peak, "ts": ts}))
            self._last_level_emit = ts

    def start(self) -> None:
        blocksize = int(self.config.sample_rate * (self.config.chunk_ms / 1000.0))
        self._stream = sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            blocksize=blocksize,
            dtype="int16",
            device=self.config.device,
            callback=self._callback,
        )
        self._stream.start()
        self.logger.info("Audio started sr=%s chunk_ms=%s device=%s blocksize=%s",
                         self.config.sample_rate, self.config.chunk_ms, self.config.device, blocksize)

    def stop(self) -> None:
        if self._stream:
            try:
                self._stream.stop()
            except Exception:
                pass
            try:
                self._stream.close()
            except Exception:
                pass
            self._stream = None
            self.logger.info("Audio stopped.")
