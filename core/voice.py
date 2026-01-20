from __future__ import annotations

import json
import queue
import time
from pathlib import Path
from typing import Any

import sounddevice as sd
from vosk import KaldiRecognizer, Model

from .config import VOSK_MODEL_DIR, VOICE_SAMPLE_RATE


class VoiceInput:
    def __init__(self, model_dir: Path | None = None, sample_rate: int | None = None, device: int | None = None) -> None:
        self.model_dir = Path(model_dir) if model_dir else VOSK_MODEL_DIR
        self.sample_rate = sample_rate or VOICE_SAMPLE_RATE
        self.device = device
        if not self.model_dir.exists():
            raise FileNotFoundError(
                f"Vosk model not found at {self.model_dir}. "
                "Download it with: python scripts/download_vosk_ru.py"
            )
        self.model = Model(str(self.model_dir))
        self.queue: queue.Queue[bytes] = queue.Queue()

    def _callback(self, indata: bytes, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        if status:
            # не глушим полностью поток — иногда драйверы дают status, но данные приходят
            pass
        self.queue.put(bytes(indata))

    def listen_once(self, timeout_sec: float = 8.0, silence_timeout_sec: float = 0.7) -> str | None:
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
