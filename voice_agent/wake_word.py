from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from openwakeword.model import Model

from .bus import Event, EventBus


@dataclass(frozen=True)
class WakeWordConfig:
    enabled: bool = True
    backend: str = "openwakeword"
    model_paths: tuple[str, ...] = ()
    model_names: tuple[str, ...] = ()
    threshold: float = 0.6
    patience_frames: int = 2
    cooldown_ms: int = 1200
    sample_rate: int = 16000
    min_rms: float = 0.01
    inference_framework: str = "onnx"
    vad_threshold: float = 0.0
    base_path: Path = Path(".")


class WakeWordDetector:
    def __init__(self, config: WakeWordConfig, bus: EventBus) -> None:
        self.config = config
        self.bus = bus
        self.logger = logging.getLogger("voice_agent")
        self._last_trigger_ts = 0.0
        self._model = None
        self._thresholds: dict[str, float] = {}
        self._patience: dict[str, int] = {}
        if self.config.enabled:
            self._load_backend()

    def _load_backend(self) -> None:
        if self.config.backend != "openwakeword":
            raise ValueError(f"Unsupported wake-word backend: {self.config.backend}")
        model_paths = self._resolve_model_paths(self.config.model_paths, self.config.model_names)
        if not model_paths:
            raise ValueError("Wake-word model_paths or model_names must be provided")
        self._model = Model(
            wakeword_models=list(model_paths),
            vad_threshold=self.config.vad_threshold,
            inference_framework=self.config.inference_framework,
        )
        self._thresholds = {name: self.config.threshold for name in self._model.models.keys()}
        if self.config.patience_frames > 0:
            self._patience = {name: self.config.patience_frames for name in self._model.models.keys()}

    def _resolve_model_paths(self, paths: Iterable[str], names: Iterable[str]) -> list[str]:
        resolved: list[str] = []
        missing: list[Path] = []
        for path in paths:
            if not path:
                continue
            candidate = Path(path)
            if not candidate.is_absolute():
                candidate = self.config.base_path / candidate
            if candidate.exists():
                resolved.append(str(candidate))
            else:
                missing.append(candidate)
        for name in names:
            if name:
                resolved.append(name)
        if not resolved and missing:
            missing_paths = ", ".join(str(path) for path in missing)
            raise ValueError(
                "Wake-word model paths not found: "
                f"{missing_paths}. Provide valid .onnx/.tflite paths or set model_names."
            )
        return resolved

    def process_chunk(self, chunk: np.ndarray, ts: float) -> None:
        if not self._model:
            return
        audio = chunk.astype(np.float32) / 32768.0
        if audio.ndim > 1:
            audio = audio[:, 0]
        if audio.size == 0:
            return
        rms = float(np.sqrt(np.mean(np.square(audio))))
        if rms < self.config.min_rms:
            return
        cooldown_s = self.config.cooldown_ms / 1000.0
        if ts - self._last_trigger_ts < cooldown_s:
            return
        predictions = self._model.predict(audio, patience=self._patience, threshold=self._thresholds)
        if any(score >= self.config.threshold for score in predictions.values()):
            self._last_trigger_ts = ts
            self.bus.publish(Event("wake_word.detected", {"ts": ts, "scores": predictions}))
