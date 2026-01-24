from __future__ import annotations

import json
import queue
import time
from pathlib import Path
from typing import Any

import sounddevice as sd
import numpy as np
from vosk import KaldiRecognizer, Model

from .config import (
    VOICE_ENGINE,
    VOICE_SAMPLE_RATE,
    VOSK_MODEL_DIR,
    WHISPER_DEVICE,
    WHISPER_MODEL_NAME,
)


class VoiceInput:
    def __init__(
        self,
        model_dir: Path | None = None,
        sample_rate: int | None = None,
        device: int | None = None,
        engine: str | None = None,
        model_size: str | None = None,
    ) -> None:
        self.sample_rate = sample_rate or VOICE_SAMPLE_RATE
        self.device = device
        self.engine = (engine or VOICE_ENGINE).strip().lower()
        self.model_size = (model_size or "").strip().lower() or None
        self.whisper_device = WHISPER_DEVICE
        if self.engine == "whisper":
            try:
                import whisper  # type: ignore
            except Exception as exc:
                raise ImportError(
                    "Whisper not installed. Install with: pip install openai-whisper"
                ) from exc
            try:
                import torch  # type: ignore
            except Exception:
                torch = None
            if self.whisper_device == "cuda" and (not torch or not torch.cuda.is_available()):
                self.whisper_device = "cpu"
            model_name = self._resolve_whisper_model_name()
            self.model = whisper.load_model(model_name, device=self.whisper_device)
        else:
            self.model_dir = Path(model_dir) if model_dir else VOSK_MODEL_DIR
            if not self.model_dir.exists():
                raise FileNotFoundError(
                    f"Vosk model not found at {self.model_dir}. "
                    "Download it with: python scripts/download_vosk_ru.py "
                    "(or set VOSK_MODEL_DIR / VOSK_MODEL_SIZE=small)."
                )
            self.model = Model(str(self.model_dir))
        self.queue: queue.Queue[bytes] = queue.Queue()

    def _resolve_whisper_model_name(self) -> str:
        # openai-whisper supports: tiny, base, small, medium, large (+ large-v2/v3)
        # Keep an allow-list so UI dropdown values work without surprises.
        allowed = {
            "tiny",
            "base",
            "small",
            "medium",
            "large",
            "large-v2",
            "large-v3",
        }
        if self.model_size in allowed:
            return self.model_size
        return WHISPER_MODEL_NAME

    def _callback(self, indata: bytes, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        if status:
            # не глушим полностью поток — иногда драйверы дают status, но данные приходят
            pass
        self.queue.put(bytes(indata))

    def listen_once(self, timeout_sec: float = 5.0, silence_timeout_sec: float = 0.5) -> str | None:
        if self.engine == "whisper":
            return self._listen_once_whisper(timeout_sec, silence_timeout_sec)
        # очистим очередь от старого мусора
        while True:
            try:
                self.queue.get_nowait()
            except queue.Empty:
                break
        recognizer = KaldiRecognizer(self.model, self.sample_rate)
        start = time.monotonic()
        last_speech_at: float | None = None
        blocksize = 1600 if self.sample_rate == 16000 else 8000
        with sd.RawInputStream(
            samplerate=self.sample_rate,
            device=self.device,
            blocksize=blocksize,
            dtype="int16",
            channels=1,
            callback=self._callback,
        ):
            while time.monotonic() - start < timeout_sec:
                try:
                    data = self.queue.get(timeout=0.2)
                except queue.Empty:
                    continue
                if recognizer.AcceptWaveform(data):
                    result = json.loads(recognizer.Result())
                    text = (result.get("text") or "").strip()
                    if text:
                        return text
                else:
                    partial = json.loads(recognizer.PartialResult())
                    partial_text = (partial.get("partial") or "").strip()
                    if partial_text:
                        last_speech_at = time.monotonic()
                if last_speech_at and time.monotonic() - last_speech_at >= silence_timeout_sec:
                    partial = json.loads(recognizer.FinalResult())
                    final_text = (partial.get("text") or "").strip()
                    return final_text or None
            partial = json.loads(recognizer.FinalResult())
            final_text = (partial.get("text") or "").strip()
            return final_text or None

    def _listen_once_whisper(self, timeout_sec: float, silence_timeout_sec: float) -> str | None:
        frames: list[np.ndarray] = []
        start = time.monotonic()
        last_voice_at: float | None = None
        blocksize = 1600 if self.sample_rate == 16000 else 8000

        def callback(indata: np.ndarray, frames_count: int, time_info: Any, status: sd.CallbackFlags) -> None:
            if status:
                pass
            frames.append(indata.copy())

        with sd.InputStream(
            samplerate=self.sample_rate,
            device=self.device,
            blocksize=blocksize,
            channels=1,
            dtype="float32",
            callback=callback,
        ):
            while time.monotonic() - start < timeout_sec:
                time.sleep(0.05)
                if not frames:
                    continue
                recent = frames[-1]
                if recent.size == 0:
                    continue
                rms = float(np.sqrt(np.mean(np.square(recent))))
                if rms > 0.01:
                    last_voice_at = time.monotonic()
                elif last_voice_at and time.monotonic() - last_voice_at >= silence_timeout_sec:
                    break

        if not frames:
            return None
        audio = np.concatenate(frames, axis=0).flatten()
        use_fp16 = self.whisper_device == "cuda"
        try:
            result = self.model.transcribe(audio, language="ru", fp16=use_fp16)
        except Exception:
            return None
        text = (result.get("text") or "").strip()
        return text or None
