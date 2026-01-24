from __future__ import annotations

"""voice_agent.wake_word

Wake-word detector with pluggable backends.

Backends:
- openwakeword: uses openWakeWord Model (custom/pretrained .onnx/.tflite)
- vosk: fast keyword spotting using Vosk partial results (grammar limited)

This module is defensive: if a backend cannot be loaded it will log an error
and disable detection instead of crashing the UI.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .bus import Event, EventBus


@dataclass(frozen=True)
class WakeWordConfig:
    enabled: bool = True
    backend: str = "openwakeword"  # "openwakeword" | "vosk"

    # openwakeword
    model_paths: tuple[str, ...] = ()
    model_names: tuple[str, ...] = ()
    threshold: float = 0.6
    patience_frames: int = 2
    inference_framework: str = "onnx"
    vad_threshold: float = 0.0

    # vosk keyword spotting
    keyword: str = "agent"
    keyword_aliases: tuple[str, ...] = ()
    vosk_model_path: str | None = None

    # common
    cooldown_ms: int = 1200
    sample_rate: int = 16000
    min_rms: float = 0.01
    base_path: Path = Path(".")


class WakeWordDetector:
    def __init__(self, config: WakeWordConfig, bus: EventBus) -> None:
        self.config = config
        self.bus = bus
        self.logger = logging.getLogger("voice_agent")

        self._last_trigger_ts = 0.0
        self._model_oww = None
        self._thresholds: dict[str, float] = {}
        self._patience: dict[str, int] = {}

        self._vosk_model = None
        self._vosk_rec = None

        if self.config.enabled:
            try:
                self._load_backend()
            except Exception as e:
                # Do not crash the entire runtime/UI because of wake backend
                self.logger.error("Wake-word backend load failed: %s", e)
                self._disable()

    def _disable(self) -> None:
        self._model_oww = None
        self._thresholds = {}
        self._patience = {}
        self._vosk_model = None
        self._vosk_rec = None

    def _load_backend(self) -> None:
        backend = (self.config.backend or "").strip().lower()

        if backend in {"openwakeword", "oww"}:
            self._load_openwakeword()
            return

        if backend in {"vosk", "keyword", "keyword_vosk"}:
            self._load_vosk()
            return

        raise ValueError(f"Unsupported wake-word backend: {self.config.backend}")

    # ---------- openwakeword ----------
    def _load_openwakeword(self) -> None:
        try:
            from openwakeword.model import Model  # type: ignore
        except Exception as e:
            raise RuntimeError("openwakeword is not installed") from e

        model_paths = self._resolve_model_paths(self.config.model_paths, self.config.model_names)
        if not model_paths:
            raise ValueError("Wake-word model_paths or model_names must be provided for openwakeword backend")

        self._model_oww = Model(
            wakeword_models=list(model_paths),
            vad_threshold=self.config.vad_threshold,
            inference_framework=self.config.inference_framework,
        )
        self._thresholds = {name: self.config.threshold for name in self._model_oww.models.keys()}
        if self.config.patience_frames > 0:
            self._patience = {name: self.config.patience_frames for name in self._model_oww.models.keys()}

        self.logger.info("Wake backend=openwakeword models=%s", list(self._model_oww.models.keys()))

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

    # ---------- vosk keyword spotting ----------
    def _load_vosk(self) -> None:
        try:
            import vosk  # type: ignore
        except Exception as e:
            raise RuntimeError("vosk is not installed") from e

        model_path = self._resolve_vosk_model_path(self.config.vosk_model_path)
        if not model_path:
            raise ValueError(
                "wake_word.vosk_model_path is required for backend=vosk. "
                "Point it to a Vosk model directory (e.g., models/vosk-model-small-ru-0.22)."
            )

        self._vosk_model = vosk.Model(str(model_path))

        # Grammar-limited recognizer = faster & fewer false words
        keywords = self._keyword_set()
        grammar = "[" + ",".join(f'"{k}"' for k in keywords) + "]"
        self._vosk_rec = vosk.KaldiRecognizer(self._vosk_model, self.config.sample_rate, grammar)
        self._vosk_rec.SetWords(False)

        self.logger.info("Wake backend=vosk keyword=%s aliases=%s", self.config.keyword, list(self.config.keyword_aliases))

    def _resolve_vosk_model_path(self, path: Optional[str]) -> Optional[Path]:
        if not path:
            return None
        p = Path(path)
        if not p.is_absolute():
            p = self.config.base_path / p
        return p if p.exists() else None

    def _keyword_set(self) -> list[str]:
        base = (self.config.keyword or "agent").strip().lower()
        out = [base]
        for a in self.config.keyword_aliases:
            aa = (a or "").strip().lower()
            if aa and aa not in out:
                out.append(aa)
        return out

    # ---------- processing ----------
    def process_chunk(self, chunk: np.ndarray, ts: float) -> None:
        if not self.config.enabled:
            return

        # common RMS gate
        audio_f = chunk.astype(np.float32)
        if audio_f.ndim > 1:
            audio_f = audio_f[:, 0]
        if audio_f.size == 0:
            return

        audio = audio_f / 32768.0
        rms = float(np.sqrt(np.mean(np.square(audio))))
        if rms < self.config.min_rms:
            return

        cooldown_s = self.config.cooldown_ms / 1000.0
        if ts - self._last_trigger_ts < cooldown_s:
            return

        backend = (self.config.backend or "").strip().lower()

        if backend in {"openwakeword", "oww"}:
            self._process_openwakeword(chunk, ts)
            return

        if backend in {"vosk", "keyword", "keyword_vosk"}:
            self._process_vosk(chunk, ts)
            return

    def _process_openwakeword(self, chunk: np.ndarray, ts: float) -> None:
        if not self._model_oww:
            return

        audio = chunk.astype(np.float32) / 32768.0
        if audio.ndim > 1:
            audio = audio[:, 0]

        predictions = self._model_oww.predict(audio, patience=self._patience, threshold=self._thresholds)
        if any(score >= self.config.threshold for score in predictions.values()):
            self._last_trigger_ts = ts
            self.bus.publish(Event("wake_word.detected", {"ts": ts, "scores": predictions}))

    def _process_vosk(self, chunk: np.ndarray, ts: float) -> None:
        if not self._vosk_rec:
            return

        # Vosk expects int16 little-endian bytes
        if chunk.dtype != np.int16:
            pcm = chunk.astype(np.int16)
        else:
            pcm = chunk

        self._vosk_rec.AcceptWaveform(pcm.tobytes())
        partial = self._vosk_rec.PartialResult() or ""

        # Very cheap substring match; grammar restricts outputs anyway
        partial_low = partial.lower()
        for kw in self._keyword_set():
            if kw and kw in partial_low:
                self._last_trigger_ts = ts
                self.logger.info("[WAKE/VOSK] Triggered keyword=%s", kw)
                self.bus.publish(Event("wake_word.detected", {"ts": ts, "scores": {kw: 1.0}}))

                # reset recognizer to avoid getting stuck in the same partial
                try:
                    self._vosk_rec.Reset()
                except Exception:
                    pass
                return
