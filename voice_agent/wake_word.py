from __future__ import annotations

import logging
import time
from collections import deque
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
    threshold: float = 0.45
    patience_frames: int = 2
    cooldown_ms: int = 1000
    sample_rate: int = 16000
    min_rms: float = 0.006
    inference_framework: str = "onnx"
    vad_threshold: float = 0.0
    base_path: Path = Path(".")
    preroll_ms: int = 450
    window_ms: int = 1000  # сколько аудио даём модели на вход


class WakeWordDetector:
    """
    Fast wake-word detector based on openwakeword.Model.
    - слушает постоянно (в IDLE)
    - при срабатывании даёт событие wake_word.detected
    - хранит preroll, чтобы ASR не отрезал начало команды
    """

    def __init__(self, config: WakeWordConfig, bus: EventBus) -> None:
        self.config = config
        self.bus = bus
        self.logger = logging.getLogger("voice_agent")

        self._model: Model | None = None
        self._cooldown_until = 0.0
        self._patience_hits = 0

        self._buffer: deque[int] = deque()
        self._preroll: np.ndarray = np.zeros((0,), dtype=np.int16)

        # Streaming buffer for openWakeWord (we feed ~80ms frames into Model.predict)
        self._oww_stream_buf = np.zeros((0,), dtype=np.int16)
        self._oww_chunk_samples = 1280  # 80ms @ 16kHz

        self._model_names: list[str] = []

        if self.config.enabled:
            self._load_backend()

    # ---------------------------
    # Public API
    # ---------------------------

    def take_preroll(self) -> np.ndarray:
        pr = self._preroll
        self._preroll = np.zeros((0,), dtype=np.int16)
        return pr

    def process_chunk(self, chunk: np.ndarray, ts: float | None = None) -> bool:
        if not self.config.enabled or self._model is None:
            return False

        now = ts if ts is not None else time.time()

        # mono
        if chunk.ndim > 1:
            chunk = chunk[:, 0]

        # to int16 PCM (openwakeword ожидает int16)
        if chunk.dtype != np.int16:
            # если float [-1..1]
            chunk = (np.clip(chunk, -1.0, 1.0) * 32767.0).astype(np.int16)

        # Добавляем для preroll-буфера (для ASR после детекта)
        self._buffer.extend(chunk.tolist())

        sr = self.config.sample_rate
        preroll_samples = int(sr * (self.config.preroll_ms / 1000.0))
        window_samples = int(sr * (self.config.window_ms / 1000.0))
        keep = preroll_samples + window_samples
        while len(self._buffer) > keep:
            self._buffer.popleft()

        # В cooldown мы всё равно продолжаем кормить openWakeWord аудио,
        # чтобы его внутреннее состояние "протекало" дальше (иначе может залипать).
        in_cooldown = now < self._cooldown_until

        # ВАЖНО: openWakeWord рассчитан на потоковое питание чанками (~80ms)
        # Если кормить модель перекрывающимися 1-сек окнами на каждом 20ms колбэке,
        # она начинает "накапливать" сигнал и легко уходит в постоянный триггер.
        self._oww_stream_buf = np.concatenate([self._oww_stream_buf, chunk])

        best_name = ""
        best_score = 0.0

        # Пакетно прогоняем доступные 80ms фреймы
        while self._oww_stream_buf.shape[0] >= self._oww_chunk_samples:
            frame = self._oww_stream_buf[: self._oww_chunk_samples]
            self._oww_stream_buf = self._oww_stream_buf[self._oww_chunk_samples :]

            # RMS gate применяем к решению (а не к подаче), чтобы не ломать стрим
            audio_f = frame.astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(audio_f * audio_f) + 1e-12))

            scores = self._model.predict(frame)  # dict{name: score}

            frame_best_name = ""
            frame_best_score = 0.0
            for name in self._model_names:
                if name in scores:
                    s = float(scores[name])
                    if s > frame_best_score:
                        frame_best_score = s
                        frame_best_name = name

            if rms < self.config.min_rms:
                frame_best_score = 0.0
                frame_best_name = ""

            best_name = frame_best_name
            best_score = frame_best_score

            # patience (нужно N подряд фреймов выше threshold)
            if in_cooldown:
                # в cooldown ничего не триггерим
                self._patience_hits = 0
            else:
                if best_score >= self.config.threshold:
                    self._patience_hits += 1
                else:
                    self._patience_hits = 0

                if self._patience_hits >= self.config.patience_frames:
                    break

        if self._patience_hits < self.config.patience_frames:
            return False

        # triggered
        self._patience_hits = 0
        self._cooldown_until = now + (self.config.cooldown_ms / 1000.0)

        # После триггера полезно сбросить внутреннее состояние openWakeWord,
        # чтобы не залипать в высоких скоринг-состояниях.
        if hasattr(self._model, "reset"):
            try:
                self._model.reset()
            except Exception:
                pass

        # сохраняем preroll
        if preroll_samples > 0 and len(self._buffer) >= preroll_samples:
            self._preroll = np.fromiter(list(self._buffer)[-preroll_samples:], dtype=np.int16)
        else:
            self._preroll = np.zeros((0,), dtype=np.int16)

        self.logger.info("[WAKE] Triggered name=%s score=%.3f", best_name, best_score)

        # событие в bus
        self.bus.publish(
            Event(
                "wake_word.detected",
                {
                    "name": best_name,
                    "score": best_score,
                    "ts": now,
                },
            )
        )
        return True

    # ---------------------------
    # Internal
    # ---------------------------

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

        # Чистое стартовое состояние
        if hasattr(self._model, "reset"):
            try:
                self._model.reset()
            except Exception:
                pass

        # какие имена выходов использовать
        if self.config.model_names:
            self._model_names = list(self.config.model_names)
        else:
            self._model_names = list(self._model.models.keys())

        self.logger.info("[WAKE] Loaded models: %s", ", ".join(self._model_names))

    def _resolve_model_paths(self, paths: Iterable[str], names: Iterable[str]) -> list[str]:
        resolved: list[str] = []
        missing: list[Path] = []

        # абсолютные/относительные пути
        for path in paths:
            if not path:
                continue
            p = Path(path)
            if not p.is_absolute():
                p = self.config.base_path / p
            if p.exists():
                resolved.append(str(p))
            else:
                missing.append(p)

        # именованные модели (если кто-то использует встроенные имена)
        for name in names:
            if name:
                resolved.append(name)

        if not resolved and missing:
            msg = ", ".join(str(x) for x in missing)
            raise ValueError(f"Wake-word model paths not found: {msg}")

        return resolved
