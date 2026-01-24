from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
from openwakeword.model import Model
try:
    import yaml
except Exception:
    yaml = None

from .bus import Event, EventBus

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = None
if (BASE_DIR / "voice_agent" / "config.yaml").exists():
    DEFAULT_CONFIG_PATH = BASE_DIR / "voice_agent" / "config.yaml"
elif (BASE_DIR / "config.yaml").exists():
    DEFAULT_CONFIG_PATH = BASE_DIR / "config.yaml"


@dataclass(frozen=True)
class WakeWordConfig:
    enabled: bool = True
    backend: str = "openwakeword"
    model_paths: tuple[str, ...] = ()
    model_names: tuple[str, ...] = ()  # если пусто — берём ключи из модели
    threshold: float = 0.6
    patience_frames: int = 2
    cooldown_ms: int = 1200
    sample_rate: int = 16000
    min_rms: float = 0.01
    inference_framework: str = "onnx"
    vad_threshold: float = 0.0
    base_path: Path = Path(".")
    preroll_ms: int = 450
    total_sec: float = 2.0


def _load_yaml(path: Path | None) -> dict:
    if yaml is None or not path or not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


class WakeWordDetector:
    def __init__(self, config: WakeWordConfig, bus: EventBus) -> None:
        self.config = config
        self.bus = bus
        self.logger = logging.getLogger("voice_agent")

        self._apply_total_sec_from_config()

        self._model: Model | None = None

        # буфер под окно + preroll
        self._buffer: deque[int] = deque()
        self._cooldown_until: float = 0.0
        self._patience_hits: int = 0
        self._preroll: np.ndarray = np.zeros((0,), dtype=np.int16)

        self._active_names: list[str] = []

        if self.config.enabled:
            self._load_backend()

    @property
    def model(self) -> Model | None:
        return self._model

    def _apply_total_sec_from_config(self) -> None:
        cfg = _load_yaml(DEFAULT_CONFIG_PATH)
        if not cfg:
            self.logger.warning(
                "[WAKE] Config not found, using default total_sec=%.2f",
                self.config.total_sec,
            )
            return

        ww = cfg.get("wake_word", {}) or {}
        total_sec = ww.get("total_sec", None)
        if total_sec is None:
            return
        try:
            total_sec = float(total_sec)
        except Exception:
            return

        if abs(total_sec - self.config.total_sec) > 1e-3:
            self.logger.info(
                "[WAKE] total_sec from config=%.2f (overriding %.2f)",
                total_sec,
                self.config.total_sec,
            )
            self.config = replace(self.config, total_sec=total_sec)

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

        # какие имена моделей реально доступны
        available = list(self._model.models.keys())
        if self.config.model_names:
            # оставляем только существующие
            self._active_names = [n for n in self.config.model_names if n in available]
            if not self._active_names:
                self._active_names = available
        else:
            self._active_names = available

        self.logger.info("[WAKE] Loaded models: %s", ", ".join(self._active_names))

    def _resolve_model_paths(self, paths: Iterable[str], names: Iterable[str]) -> list[str]:
        resolved: list[str] = []
        missing: list[Path] = []

        # явные пути
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

        # model_names можно использовать как "builtin"/идентификатор (если у тебя так сделано)
        for name in names:
            if name:
                resolved.append(name)

        if not resolved and missing:
            raise ValueError(
                "Wake-word model paths not found: "
                + ", ".join(str(p) for p in missing)
            )

        return resolved

    def take_preroll(self) -> np.ndarray:
        pr = self._preroll
        self._preroll = np.zeros((0,), dtype=np.int16)
        return pr

    def process_chunk(self, chunk: np.ndarray, ts: float | None = None) -> bool:
        """
        Возвращает True если детектнуло wake-word.
        И СРАЗУ публикует событие wake_word.detected в bus.
        """
        if not self.config.enabled or self._model is None:
            return False

        now = ts if ts is not None else time.time()

        # mono + int16
        if chunk.ndim > 1:
            chunk = chunk[:, 0]
        if chunk.dtype != np.int16:
            # если float32 -1..1
            chunk = (np.clip(chunk, -1, 1) * 32767).astype(np.int16)

        # rms gate
        audio_f = chunk.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(audio_f * audio_f) + 1e-12))
        if rms < self.config.min_rms:
            self._patience_hits = 0
            self.logger.debug("[WAKE] skip (low rms) -> reset patience")
            return False

        # буфер
        self._buffer.extend(chunk.tolist())

        sr = self.config.sample_rate
        window_samples = int(sr * self.config.total_sec)
        preroll_samples = int(sr * (self.config.preroll_ms / 1000.0))
        max_keep = window_samples + preroll_samples

        while len(self._buffer) > max_keep:
            self._buffer.popleft()

        # cooldown
        if now < self._cooldown_until:
            return False

        if len(self._buffer) < window_samples:
            return False

        window = np.array(list(self._buffer)[-window_samples:], dtype=np.int16)

        scores = self._model.predict(window)

        max_score = 0.0
        best_name = None
        for name in self._active_names:
            if name in scores:
                s = float(scores[name])
                if s > max_score:
                    max_score = s
                    best_name = name

        # patience
        if max_score >= self.config.threshold:
            self._patience_hits += 1
        else:
            self._patience_hits = 0

        if self._patience_hits < self.config.patience_frames:
            return False

        # TRIGGER
        self._patience_hits = 0
        self._cooldown_until = now + (self.config.cooldown_ms / 1000.0)

        if preroll_samples > 0:
            self._preroll = np.array(list(self._buffer)[-preroll_samples:], dtype=np.int16)
        else:
            self._preroll = np.zeros((0,), dtype=np.int16)

        self.logger.info("[WAKE] Triggered name=%s score=%.3f", best_name, max_score)

        self.bus.publish(
            Event("wake_word.detected", {"name": best_name, "score": max_score, "ts": now})
        )
        return True
