from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .actions import ActionExecutor
from .audio_capture import AudioCapture, AudioConfig
from .asr_whisper import AsrConfig, FasterWhisperASR
from .bus import Event, EventBus
from .intent import IntentRecognizer
from .tts import TtsConfig, TtsEngine
from .vad import SileroVAD, VadConfig


@dataclass
class State:
    name: str = "IDLE"


def _load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def main() -> None:
    config_path = Path(__file__).with_name("config.yaml")
    cfg = _load_config(config_path)

    logging.basicConfig(level=cfg.get("logging", {}).get("level", "info").upper())
    logger = logging.getLogger("voice_agent")

    bus = EventBus()
    state = State()

    audio_cfg = cfg.get("audio", {})
    vad_cfg = cfg.get("vad", {})
    asr_cfg = cfg.get("asr", {})
    tts_cfg = cfg.get("tts", {})
    nlu_cfg = cfg.get("nlu", {})

    audio = AudioCapture(
        AudioConfig(
            sample_rate=audio_cfg.get("sample_rate", 16000),
            channels=audio_cfg.get("channels", 1),
            chunk_ms=audio_cfg.get("chunk_ms", 20),
            device=audio_cfg.get("device"),
        ),
        bus,
    )
    vad = SileroVAD(
        VadConfig(
            device=vad_cfg.get("device", "cpu"),
            threshold=vad_cfg.get("threshold", 0.5),
            min_speech_ms=vad_cfg.get("min_speech_ms", 200),
            end_silence_ms=vad_cfg.get("end_silence_ms", 500),
            sample_rate=audio_cfg.get("sample_rate", 16000),
        ),
        bus,
    )
    asr = FasterWhisperASR(
        AsrConfig(
            model=asr_cfg.get("model", "small"),
            device=asr_cfg.get("device", "cuda"),
            compute_type=asr_cfg.get("compute_type", "float16"),
            beam_size=asr_cfg.get("beam_size", 2),
            language=asr_cfg.get("language", "ru"),
            max_utterance_s=asr_cfg.get("max_utterance_s", 10),
            partial_interval_ms=asr_cfg.get("partial_interval_ms", 150),
            partial_min_delta=asr_cfg.get("partial_min_delta", 3),
        ),
        bus,
    )
    intent = IntentRecognizer(nlu_cfg.get("synonyms", {}))
    actions = ActionExecutor(bus)
    tts = TtsEngine(
        TtsConfig(
            enabled=tts_cfg.get("enabled", False),
            voice=tts_cfg.get("voice", "male"),
            engine=tts_cfg.get("engine", "piper"),
        ),
        bus,
    )

    def on_audio(event: Event) -> None:
        data = event.payload["data"]
        ts = event.payload["ts"]
        vad.process_chunk(data, ts)
        asr.accept_audio(data, ts)

    def on_vad_start(event: Event) -> None:
        logger.info("VAD speech_start")
        state.name = "LISTENING"
        asr.speech_start()
        tts.stop()

    def on_vad_end(event: Event) -> None:
        logger.info("VAD speech_end")
        state.name = "DECODING"
        asr.speech_end(event.payload["ts"])

    def on_partial(event: Event) -> None:
        logger.info("ASR partial: %s", event.payload["text"])

    def on_final(event: Event) -> None:
        text = event.payload["text"]
        logger.info("ASR final: %s", text)
        state.name = "FINALIZING"
        recognized = intent.recognize(text)
        if recognized:
            bus.publish(Event("agent.intent", {"name": recognized.name, "slots": recognized.slots}))
            result = actions.run(recognized)
            logger.info("Action: %s", result.message)
            tts.speak("Готово")
        else:
            logger.info("Intent: not understood")
        state.name = "IDLE"

    bus.subscribe("audio.chunk", on_audio)
    bus.subscribe("vad.speech_start", on_vad_start)
    bus.subscribe("vad.speech_end", on_vad_end)
    bus.subscribe("asr.partial", on_partial)
    bus.subscribe("asr.final", on_final)
    bus.subscribe("action.run", lambda e: logger.info("Action run: %s", e.payload))
    bus.subscribe("agent.intent", lambda e: logger.info("Intent: %s", e.payload))

    audio.start()

    def loop() -> None:
        while True:
            event = bus.poll(timeout=0.1)
            if event:
                bus.dispatch(event)

    thread = threading.Thread(target=loop, daemon=True)
    thread.start()
    logger.info("Voice agent started.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        logger.info("Stopping...")
        audio.stop()


if __name__ == "__main__":
    main()
