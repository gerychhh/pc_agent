from __future__ import annotations

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
        self._stream: sd.InputStream | None = None

    def _callback(self, indata: np.ndarray, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        if status:
            pass
        timestamp = time.monotonic()
        payload = {"data": indata.copy(), "ts": timestamp}
        self.bus.publish(Event("audio.chunk", payload))

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
