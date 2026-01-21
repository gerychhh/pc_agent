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
    def __init__(self, config: AudioConfig, bus: EventBus) -> None:
        self.config = config
        self.bus = bus
        self.logger = logging.getLogger("voice_agent")
        self._stream: sd.InputStream | None = None
        self._last_level_emit = 0.0
        self._muted = False

    def set_muted(self, muted: bool) -> None:
        self._muted = muted

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        if status:
            self.logger.warning("Audio stream status: %s", status)
        timestamp = time.monotonic()
        payload = {"data": indata.copy(), "ts": timestamp}
        if not self._muted:
            self.bus.publish(Event("audio.chunk", payload))
        if timestamp - self._last_level_emit >= 0.5:
            rms = float(np.sqrt(np.mean(indata.astype(np.float32) ** 2))) if indata.size else 0.0
            self.bus.publish(Event("audio.level", {"rms": rms, "ts": timestamp}))
            self._last_level_emit = timestamp

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

    def stop(self) -> None:
        if self._stream:
            self._stream.stop()
            self._stream.close()
            self._stream = None
