from __future__ import annotations

"""voice_agent.wake_word

Wake-word detector with pluggable backends.

Backends:
- openwakeword: uses openWakeWord Model (.onnx/.tflite) and emits continuous scores like scripts/test_beavis.py
- vosk: keyword spotting using Vosk grammar-limited partial results

This module is defensive: if a backend cannot be loaded it will log an error
and disable detection instead of crashing the UI/runtime.

Events:
  - wake_word.scores: {"ts": float, "scores": {name: prob}, "best": float, "best_name": str,
                       "timeline": str, "bar": int, "char": str}
  - wake_word.detected: {"ts": float, "scores": {...}, "best": float, "best_name": str}
"""

import logging
import time
from collections import deque
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

    # visualization
    history_size: int = 40


class WakeWordDetector:
    def __init__(self, config: WakeWordConfig, bus: EventBus) -> None:
        self.config = config
        self.bus = bus
        self.logger = logging.getLogger("voice_agent.wake")

        self._last_trigger_ts = 0.0

        # openwakeword state
        self._model_oww = None
        self._oww_buf = np.zeros(0, dtype=np.int16)
        self._oww_frame = 1280  # 80ms at 16kHz (matches scripts/test_beavis.py)
        self._patience_left: dict[str, int] = {}
        self._patience_required: int = max(1, int(self.config.patience_frames or 1))

        # visualization timeline (like test_beavis.py)
        self._timeline = deque(['-'] * max(10, int(self.config.history_size)), maxlen=max(10, int(self.config.history_size)))

        # vosk state
        self._vosk_model = None
        self._vosk_rec = None

        if self.config.enabled:
            try:
                self._load_backend()
            except Exception as e:
                self.logger.error("Wake-word backend load failed: %s", e)
                self.logger.error("Wake-word detection DISABLED. Fix wake_word.model_paths/model_names in voice_agent/config.yaml")
                self._disable()

    def _disable(self) -> None:
        self._model_oww = None
        self._oww_buf = np.zeros(0, dtype=np.int16)
        self._patience_left = {}
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
            vad_threshold=float(self.config.vad_threshold or 0.0),
            inference_framework=self.config.inference_framework,
        )
        self._patience_left = {name: self._patience_required for name in self._model_oww.models.keys()}
        self.logger.info("Wake backend=openwakeword models=%s threshold=%.3f patience=%d cooldown_ms=%d",
                         list(self._model_oww.models.keys()), self.config.threshold, self._patience_required, self.config.cooldown_ms)

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
            # openwakeword has built-in names too; allow as-is
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
                "Point it to a Vosk model directory (e.g., ../models/vosk-model-small-ru-0.22)."
            )

        self._vosk_model = vosk.Model(str(model_path))

        keywords = self._keyword_set()
        grammar = "[" + ",".join(f'"{k}"' for k in keywords) + "]"
        self._vosk_rec = vosk.KaldiRecognizer(self._vosk_model, self.config.sample_rate, grammar)
        self._vosk_rec.SetWords(False)

        self.logger.info("Wake backend=vosk keyword=%s aliases=%s model=%s", self.config.keyword, list(self.config.keyword_aliases), model_path)

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

        backend = (self.config.backend or "").strip().lower()
        if backend in {"openwakeword", "oww"}:
            self._process_openwakeword(chunk, ts)
            return
        if backend in {"vosk", "keyword", "keyword_vosk"}:
            self._process_vosk(chunk, ts)
            return

    def _emit_scores(self, ts: float, scores: dict[str, float]) -> None:
        if not scores:
            return
        best_name = max(scores, key=lambda k: scores[k])
        best = float(scores[best_name])

        # timeline char mapping like scripts/test_beavis.py
        if best > 0.99:
            char = "█"
        elif best > 0.80:
            char = "▓"
        elif best > 0.70:
            char = "▒"
        elif best > 0.10:
            char = "░"
        else:
            char = "·"

        self._timeline.append(char)
        bar = int(max(0.0, min(1.0, best)) * 20)

        self.bus.publish(
            Event(
                "wake_word.scores",
                {
                    "ts": ts,
                    "scores": scores,
                    "best": best,
                    "best_name": best_name,
                    "timeline": "".join(self._timeline),
                    "bar": bar,
                    "char": char,
                },
            )
        )

    def _process_openwakeword(self, chunk: np.ndarray, ts: float) -> None:
        if not self._model_oww:
            return

        # Ensure int16 mono
        if chunk.ndim > 1:
            chunk = chunk[:, 0]
        if chunk.dtype != np.int16:
            chunk = chunk.astype(np.int16, copy=False)

        # Buffer into 80ms frames (1280 samples) like test_beavis.py
        if chunk.size:
            self._oww_buf = np.concatenate([self._oww_buf, chunk], axis=0)

        cooldown_s = self.config.cooldown_ms / 1000.0
        if ts - self._last_trigger_ts < cooldown_s:
            # still emit scores for UI (but do not trigger)
            while self._oww_buf.size >= self._oww_frame:
                frame = self._oww_buf[: self._oww_frame]
                self._oww_buf = self._oww_buf[self._oww_frame :]
                scores = {k: float(v) for k, v in (self._model_oww.predict(frame) or {}).items()}
                self._emit_scores(ts, scores)
            return

        while self._oww_buf.size >= self._oww_frame:
            frame = self._oww_buf[: self._oww_frame]
            self._oww_buf = self._oww_buf[self._oww_frame :]

            # RMS gate (normalized)
            audio_f = frame.astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(audio_f * audio_f))) if audio_f.size else 0.0
            if rms < float(self.config.min_rms or 0.0):
                continue

            pred = self._model_oww.predict(frame) or {}
            scores = {k: float(v) for k, v in pred.items()}
            self._emit_scores(ts, scores)

            # trigger logic with patience
            for name, score in scores.items():
                if score >= float(self.config.threshold):
                    self._patience_left[name] = self._patience_left.get(name, self._patience_required) - 1
                    if self._patience_left[name] <= 0:
                        self._last_trigger_ts = ts
                        # reset patience
                        self._patience_left = {k: self._patience_required for k in self._patience_left.keys()}
                        self.bus.publish(Event("wake_word.detected", {"ts": ts, "scores": scores, "best": score, "best_name": name}))
                        return
                else:
                    self._patience_left[name] = self._patience_required

    def _process_vosk(self, chunk: np.ndarray, ts: float) -> None:
        if not self._vosk_rec:
            return

        if chunk.ndim > 1:
            chunk = chunk[:, 0]
        if chunk.dtype != np.int16:
            chunk = chunk.astype(np.int16, copy=False)

        # basic RMS gate
        x = chunk.astype(np.float32) / 32768.0
        rms = float(np.sqrt(np.mean(x * x))) if x.size else 0.0
        if rms < float(self.config.min_rms or 0.0):
            return

        cooldown_s = self.config.cooldown_ms / 1000.0
        if ts - self._last_trigger_ts < cooldown_s:
            return

        self._vosk_rec.AcceptWaveform(chunk.tobytes())
        partial = self._vosk_rec.PartialResult() or ""
        partial_low = partial.lower()

        # Emit a fake score for UI (1.0 if keyword present else 0.0)
        best = 0.0
        best_name = ""
        for kw in self._keyword_set():
            if kw and kw in partial_low:
                best = 1.0
                best_name = kw
                break
        if best_name:
            self._emit_scores(ts, {best_name: best})
            self._last_trigger_ts = ts
            self.bus.publish(Event("wake_word.detected", {"ts": ts, "scores": {best_name: 1.0}, "best": 1.0, "best_name": best_name}))
            try:
                self._vosk_rec.Reset()
            except Exception:
                pass
        else:
            # still emit 0-ish so UI shows activity
            self._emit_scores(ts, {self.config.keyword: 0.0})
