from __future__ import annotations

import json
import queue
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import sounddevice as sd
from vosk import KaldiRecognizer, Model

from .config import (
    VOICE_SAMPLE_RATE,
    VOICE_DEVICE,
    VOSK_MODEL_DIR,
    WHISPER_DEVICE,
    WHISPER_MODEL_NAME,
    WHISPER_MODEL_SIZE,
)


@dataclass(frozen=True)
class WakeGateConfig:
    """Параметры wake->command пайплайна (без кастомной wake-word модели).

    Идея: дешёво слушаем Vosk и ищем слово 'Бивис' в partial,
    а Whisper запускаем только после детекта.
    """

    wake_name: str = "Бивис"
    sample_rate: int = VOICE_SAMPLE_RATE
    device: int | None = None

    # Детект wake
    wake_timeout_sec: float = 0.0  # 0 = бесконечно
    wake_min_rms: float = 0.007  # отсечь тишину/шум

    # Запись команды
    pre_roll_sec: float = 0.5      # сколько аудио сохранить ДО детекта (чтобы не терять фразу)
    max_command_sec: float = 8.0   # максимум длины команды после детекта
    end_silence_sec: float = 0.7   # пауза тишины => конец команды
    command_min_rms: float = 0.01  # порог "голоса" для таймера тишины

    # Whisper
    whisper_model_size: str = WHISPER_MODEL_SIZE
    whisper_model_name: str = WHISPER_MODEL_NAME
    whisper_device: str = WHISPER_DEVICE
    language: str = "ru"


class WakeGatedVoiceInput:
    """VoiceInput совместимый интерфейс: listen_once() -> текст.

    Возвращает результат Whisper после wake. Если wake не найден/таймаут — None.
    """

    def __init__(
        self,
        *,
        cfg: WakeGateConfig | None = None,
        model_dir: Path | None = None,
    ) -> None:
        self.cfg = cfg or WakeGateConfig()
        self.sample_rate = self.cfg.sample_rate
        self.device = self.cfg.device
        if self.device is None and VOICE_DEVICE:
            # core.config VOICE_DEVICE хранит как строку (иногда индекс)
            try:
                self.device = int(VOICE_DEVICE)
            except Exception:
                self.device = None

        self.vosk_dir = Path(model_dir) if model_dir else VOSK_MODEL_DIR
        if not self.vosk_dir.exists():
            raise FileNotFoundError(
                f"Vosk model not found at {self.vosk_dir}. "
                "Download it with: python scripts/download_vosk_ru.py "
                "(or set VOSK_MODEL_DIR / VOSK_MODEL_SIZE=small)."
            )
        self.vosk_model = Model(str(self.vosk_dir))

        # Whisper загрузим один раз
        try:
            import whisper  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError("Whisper not installed. Install with: pip install openai-whisper") from exc
        try:
            import torch  # type: ignore
        except Exception:
            torch = None

        whisper_device = (self.cfg.whisper_device or "cpu").lower()
        if whisper_device == "cuda" and (not torch or not torch.cuda.is_available()):
            whisper_device = "cpu"
        self.whisper_device = whisper_device

        model_name = self.cfg.whisper_model_size.strip().lower() if self.cfg.whisper_model_size else ""
        if model_name not in {"tiny", "base", "small", "medium", "large"}:
            model_name = self.cfg.whisper_model_name
        self.whisper_model = whisper.load_model(model_name, device=self.whisper_device)

        self._queue: queue.Queue[bytes] = queue.Queue()

    def _callback(self, indata: bytes, frames: int, time_info: Any, status: sd.CallbackFlags) -> None:
        if status:
            # статус не всегда критичный
            pass
        self._queue.put(bytes(indata))

    @staticmethod
    def _rms_int16(raw: bytes) -> float:
        if not raw:
            return 0.0
        a = np.frombuffer(raw, dtype=np.int16).astype(np.float32)
        if a.size == 0:
            return 0.0
        a /= 32768.0
        return float(np.sqrt(np.mean(np.square(a))))

    def _wake_in_text(self, text: str) -> bool:
        wake = (self.cfg.wake_name or "").strip().lower()
        if not wake:
            return True
        t = (text or "").strip().lower()
        if not t:
            return False
        # мягко: ищем как отдельное слово, но без сложного NLP
        # (vosk возвращает уже токены словами)
        tokens = t.split()
        return wake in tokens or (len(wake.split()) > 1 and wake in t)

    def listen_once(self) -> str | None:
        # очистим очередь
        while True:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break

        blocksize = 1600 if self.sample_rate == 16000 else 8000  # ~0.1s для 16k
        pre_roll_chunks = max(1, int(self.cfg.pre_roll_sec / (blocksize / self.sample_rate)))
        ring: deque[bytes] = deque(maxlen=pre_roll_chunks)

        recognizer = KaldiRecognizer(self.vosk_model, self.sample_rate)

        start = time.monotonic()
        wake_found = False
        record_chunks: list[bytes] = []
        record_start: float | None = None
        last_voice_at: float | None = None

        with sd.RawInputStream(
            samplerate=self.sample_rate,
            device=self.device,
            blocksize=blocksize,
            dtype="int16",
            channels=1,
            callback=self._callback,
        ):
            while True:
                if self.cfg.wake_timeout_sec and (time.monotonic() - start) > self.cfg.wake_timeout_sec:
                    return None
                try:
                    data = self._queue.get(timeout=0.2)
                except queue.Empty:
                    continue

                ring.append(data)

                if not wake_found:
                    # отсечём тишину по RMS, чтобы не пихать мусор в vosk
                    if self._rms_int16(data) < self.cfg.wake_min_rms:
                        continue

                    # Vosk: partial и final
                    if recognizer.AcceptWaveform(data):
                        res = json.loads(recognizer.Result() or "{}")
                        txt = (res.get("text") or "").strip()
                        if self._wake_in_text(txt):
                            wake_found = True
                    else:
                        part = json.loads(recognizer.PartialResult() or "{}")
                        ptxt = (part.get("partial") or "").strip()
                        if self._wake_in_text(ptxt):
                            wake_found = True

                    if wake_found:
                        # начинаем запись команды с pre-roll, чтобы "Бивис открой ..." не обрезалось
                        record_chunks = list(ring)
                        record_start = time.monotonic()
                        last_voice_at = record_start
                        continue

                # RECORDING
                if wake_found:
                    record_chunks.append(data)
                    now = time.monotonic()
                    rms = self._rms_int16(data)
                    if rms >= self.cfg.command_min_rms:
                        last_voice_at = now
                    if record_start and (now - record_start) >= self.cfg.max_command_sec:
                        break
                    if last_voice_at and (now - last_voice_at) >= self.cfg.end_silence_sec:
                        break

        if not record_chunks:
            return None

        audio_i16 = np.frombuffer(b"".join(record_chunks), dtype=np.int16).astype(np.float32) / 32768.0
        if audio_i16.size == 0:
            return None

        use_fp16 = self.whisper_device == "cuda"
        try:
            result = self.whisper_model.transcribe(audio_i16, language=self.cfg.language, fp16=use_fp16)
        except Exception:
            return None
        text = (result.get("text") or "").strip()
        return text or None
