from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml

from .actions import ActionExecutor
from .audio_capture import AudioCapture, AudioConfig
from .asr_whisper import AsrConfig, FasterWhisperASR
from .bus import Event, EventBus
from .intent import IntentRecognizer
from .tts import TtsConfig, TtsEngine
from .vad import SileroVAD, VadConfig
from .wake_word import WakeWordConfig, WakeWordDetector


@dataclass
class State:
    name: str = "IDLE"


def _load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


class VoiceAgentRuntime:
    def __init__(
        self,
        config_path: Path | None = None,
        *,
        on_final: Callable[[str], None] | None = None,
        on_partial: Callable[[str], None] | None = None,
        on_audio_level: Callable[[float], None] | None = None,
        enable_actions: bool = True,
    ) -> None:
        self.config_path = config_path or Path(__file__).with_name("config.yaml")
        self.cfg = _load_config(self.config_path)
        logging.basicConfig(level=self.cfg.get("logging", {}).get("level", "info").upper())
        self.logger = logging.getLogger("voice_agent")
        logging.getLogger("faster_whisper").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        # torchaudio/torio может пытаться загрузить FFmpeg-расширения и спамить трейсами в DEBUG
        logging.getLogger("torio").setLevel(logging.ERROR)
        logging.getLogger("torchaudio").setLevel(logging.ERROR)
        self.bus = EventBus()
        self.state = State()
        self._thread: threading.Thread | None = None
        self._running = threading.Event()
        self._on_final_cb = on_final
        self._on_partial_cb = on_partial
        self._on_audio_level_cb = on_audio_level
        self._enable_actions = enable_actions

        audio_cfg = self.cfg.get("audio", {})
        vad_cfg = self.cfg.get("vad", {})
        asr_cfg = self.cfg.get("asr", {})
        tts_cfg = self.cfg.get("tts", {})
        nlu_cfg = self.cfg.get("nlu", {})
        wake_cfg = self.cfg.get("wake_word", {})

        # ✅ Единый источник истины для длины окна анализа wake-word
        # total_sec хранится в config.yaml (как в тренировке/тестах),
        # а детектору нужен window_ms.
        wake_total_sec = float(wake_cfg.get("total_sec", 2.0))
        wake_window_ms = int(wake_total_sec * 1000.0)
        # если кто-то хочет задавать window_ms напрямую — поддержим и это
        wake_window_ms = int(wake_cfg.get("window_ms", wake_window_ms))

        self.audio = AudioCapture(
            AudioConfig(
                sample_rate=audio_cfg.get("sample_rate", 16000),
                channels=audio_cfg.get("channels", 1),
                chunk_ms=audio_cfg.get("chunk_ms", 20),
                device=audio_cfg.get("device"),
            ),
            self.bus,
        )
        self.wake_word = WakeWordDetector(
            WakeWordConfig(
                enabled=wake_cfg.get("enabled", True),
                backend=wake_cfg.get("backend", "openwakeword"),
                model_paths=tuple(wake_cfg.get("model_paths", [])),
                model_names=tuple(wake_cfg.get("model_names", [])),
                threshold=wake_cfg.get("threshold", 0.6),
                patience_frames=wake_cfg.get("patience_frames", 2),
                cooldown_ms=wake_cfg.get("cooldown_ms", 1200),
                sample_rate=audio_cfg.get("sample_rate", 16000),
                min_rms=wake_cfg.get("min_rms", 0.01),
                inference_framework=wake_cfg.get("inference_framework", "onnx"),
                vad_threshold=wake_cfg.get("vad_threshold", 0.0),
                base_path=self.config_path.parent,
                preroll_ms=wake_cfg.get("preroll_ms", 450),
                window_ms=wake_window_ms,
            ),
            self.bus,
        )
        self.vad = SileroVAD(
            VadConfig(
                device=vad_cfg.get("device", "cpu"),
                threshold=vad_cfg.get("threshold", 0.5),
                min_speech_ms=vad_cfg.get("min_speech_ms", 300),
                end_silence_ms=vad_cfg.get("end_silence_ms", 700),
                sample_rate=audio_cfg.get("sample_rate", 16000),
                min_rms=vad_cfg.get("min_rms", 0.01),
                noise_floor_alpha=vad_cfg.get("noise_floor_alpha", 0.05),
                noise_ratio=vad_cfg.get("noise_ratio", 1.5),
            ),
            self.bus,
        )
        self.asr = FasterWhisperASR(
            AsrConfig(
                model=asr_cfg.get("model", "small"),
                device=asr_cfg.get("device", "cuda"),
                compute_type=asr_cfg.get("compute_type", "float16"),
                beam_size=asr_cfg.get("beam_size", 1),
                language=asr_cfg.get("language", "ru"),
                max_utterance_s=asr_cfg.get("max_utterance_s", 10),
                partial_interval_ms=asr_cfg.get("partial_interval_ms", 220),
                partial_min_delta=asr_cfg.get("partial_min_delta", 3),
                min_partial_s=asr_cfg.get("min_partial_s", 0.5),
                sample_rate=audio_cfg.get("sample_rate", 16000),
                no_speech_threshold=asr_cfg.get("no_speech_threshold", 0.8),
                log_prob_threshold=asr_cfg.get("log_prob_threshold", -1.0),
                compression_ratio_threshold=asr_cfg.get("compression_ratio_threshold", 2.4),
                min_buffer_s=asr_cfg.get("min_buffer_s", 0.6),
            ),
            self.bus,
        )
        self.intent = IntentRecognizer(nlu_cfg.get("synonyms", {}))
        self.actions = ActionExecutor(self.bus)
        self.tts = TtsEngine(
            TtsConfig(
                enabled=tts_cfg.get("enabled", False),
                voice=tts_cfg.get("voice", "male"),
                engine=tts_cfg.get("engine", "piper"),
            ),
            self.bus,
        )

        self.bus.subscribe("audio.chunk", self._on_audio)
        self.bus.subscribe("audio.level", self._on_audio_level)
        self.bus.subscribe("wake_word.detected", self._on_wake_word)
        self.bus.subscribe("vad.speech_start", self._on_vad_start)
        self.bus.subscribe("vad.speech_end", self._on_vad_end)
        self.bus.subscribe("asr.partial", self._on_partial)
        self.bus.subscribe("asr.final", self._on_final)
        self.bus.subscribe("action.run", lambda e: self.logger.info("Action run: %s", e.payload))
        self.bus.subscribe("agent.intent", lambda e: self.logger.info("Intent: %s", e.payload))

    def _on_audio(self, event: Event) -> None:
        data = event.payload["data"]
        ts = event.payload["ts"]
        if self.state.name == "IDLE":
            self.wake_word.process_chunk(data, ts)
            return
        if self.state.name in {"ARMED", "LISTENING"}:
            self.vad.process_chunk(data, ts)
            self.asr.accept_audio(data, ts)

    def _on_audio_level(self, event: Event) -> None:
        rms = event.payload["rms"]
        self.logger.debug("Audio RMS: %.4f", rms)
        if self._on_audio_level_cb:
            self._on_audio_level_cb(rms)

    def _on_vad_start(self, _event: Event) -> None:
        self.logger.info("VAD speech_start")
        self.state.name = "LISTENING"
        self.asr.speech_start()
        self.tts.stop()

    def _on_vad_end(self, event: Event) -> None:
        self.logger.info("VAD speech_end")
        self.state.name = "DECODING"
        self.asr.speech_end(event.payload["ts"])
        self.state.name = "IDLE"

    def _on_partial(self, event: Event) -> None:
        text = event.payload["text"]
        self.logger.debug("ASR partial: %s", text)
        if self._on_partial_cb:
            self._on_partial_cb(text)

    def _on_final(self, event: Event) -> None:
        text = event.payload["text"]
        normalized = self.intent.normalize(text)
        self.logger.info("ASR final: %s", text)
        self.logger.info("ASR normalized: %s", normalized)
        self.state.name = "FINALIZING"
        if self._on_final_cb:
            self._on_final_cb(text)
        if self._enable_actions:
            recognized = self.intent.recognize(text)
            if recognized:
                self.bus.publish(Event("agent.intent", {"name": recognized.name, "slots": recognized.slots}))
                result = self.actions.run(recognized)
                self.logger.info("Action: %s", result.message)
                self.tts.speak("Готово")
            else:
                self.logger.info("Intent: not understood (normalized=%s)", normalized)
        self.state.name = "IDLE"

    def _on_wake_word(self, event: Event) -> None:
        if self.state.name != "IDLE":
            return

        self.logger.info("Wake-word detected")
        self.state.name = "ARMED"

        # ✅ ЗАБИРАЕМ pre-roll (кусок до wake)
        preroll = self.wake_word.take_preroll()

        # ✅ И СРАЗУ КИДАЕМ ЕГО В VAD + ASR, чтобы не терять начало команды
        if preroll is not None and len(preroll) > 0:
            ts = event.payload.get("ts", time.time())
            self.vad.process_chunk(preroll, ts)
            self.asr.accept_audio(preroll, ts)

    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self.audio.start()

        def loop() -> None:
            while self._running.is_set():
                event = self.bus.poll(timeout=0.1)
                if event:
                    self.bus.dispatch(event)

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()
        self.logger.info("Voice agent started.")

    def stop(self) -> None:
        if not self._running.is_set():
            return
        self._running.clear()
        self.audio.stop()
        self.logger.info("Voice agent stopped.")


def main() -> None:
    runtime = VoiceAgentRuntime()
    runtime.start()
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        runtime.stop()


if __name__ == "__main__":
    main()
