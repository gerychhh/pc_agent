from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import yaml
import numpy as np

from .actions import ActionExecutor
from .audio_capture import AudioCapture, AudioConfig
from .asr_whisper import AsrConfig, FasterWhisperASR
from .asr_vosk import AsrVoskConfig, VoskASR
from .bus import Event, EventBus
from .intent import IntentRecognizer
from .tts import TtsConfig, TtsEngine
from .vad import SileroVAD, VadConfig
from .wake_word import WakeWordConfig, WakeWordDetector


@dataclass
class State:
    name: str = "IDLE"  # IDLE -> ARMED -> LISTENING -> DECODING
    since: float = 0.0


def _load_config(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    return data or {}


def _resolve_path(base: Path, maybe_path: str | None) -> Path | None:
    if not maybe_path:
        return None
    p = Path(maybe_path)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p


class VoiceAgentRuntime:
    """Wake-word -> VAD -> ASR runtime.

    Main callbacks:
      - on_final(text)
      - on_partial(text)
      - on_audio_level(rms)
      - on_status(kind, payload)  # for UI + console diagnostics
    """

    def __init__(
        self,
        config_path: Path | None = None,
        *,
        on_final: Callable[[str], None] | None = None,
        on_partial: Callable[[str], None] | None = None,
        on_audio_level: Callable[[float], None] | None = None,
        on_status: Callable[[str, dict[str, Any]], None] | None = None,
        enable_actions: bool = True,
    ) -> None:
        self.config_path = config_path or Path(__file__).with_name("config.yaml")
        self.cfg = _load_config(self.config_path)

        log_level = str(self.cfg.get("logging", {}).get("level", "info")).upper()
        logging.basicConfig(
            level=getattr(logging, log_level, logging.INFO),
            format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%H:%M:%S",
            force=True,
        )
        self.logger = logging.getLogger("voice_agent.runtime")
        logging.getLogger("faster_whisper").setLevel(logging.WARNING)
        # Keep noisy native modules quiet even when logging.level=debug
        logging.getLogger("torio").setLevel(logging.WARNING)
        logging.getLogger("torio._extension").setLevel(logging.WARNING)
        logging.getLogger("torio._extension.utils").setLevel(logging.WARNING)
        logging.getLogger("torchaudio").setLevel(logging.WARNING)
        logging.getLogger("sounddevice").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)

        self.bus = EventBus()
        self.state = State(name="IDLE", since=time.monotonic())

        self._thread: threading.Thread | None = None
        self._running = threading.Event()

        self._on_final_cb = on_final
        self._on_partial_cb = on_partial
        self._on_audio_level_cb = on_audio_level
        self._on_status_cb = on_status

        self._enable_actions = enable_actions

        audio_cfg = self.cfg.get("audio", {})
        vad_cfg = self.cfg.get("vad", {})
        asr_cfg = self.cfg.get("asr", {})
        tts_cfg = self.cfg.get("tts", {})
        nlu_cfg = self.cfg.get("nlu", {})
        wake_cfg = self.cfg.get("wake_word", {})

        self._armed_timeout_ms = int(wake_cfg.get("armed_timeout_ms", 6000))
        self._decode_timeout_ms = int(asr_cfg.get("decode_timeout_ms", 2500))
        self._decode_started_at: float | None = None

        self._preroll_ms = int(wake_cfg.get("preroll_ms", 450))
        self._preroll_samples = max(0, int(int(audio_cfg.get("sample_rate", 16000)) * (self._preroll_ms / 1000.0)))
        self._preroll_buf = np.zeros(0, dtype=np.int16)

        self._last_chunk: Any = None
        self._last_chunk_ts: float = 0.0
        self._armed_since: float | None = None

        # Components
        self.audio = AudioCapture(
            AudioConfig(
                sample_rate=int(audio_cfg.get("sample_rate", 16000)),
                channels=int(audio_cfg.get("channels", 1)),
                chunk_ms=int(audio_cfg.get("chunk_ms", 20)),
                device=audio_cfg.get("device"),
            ),
            self.bus,
        )

        self.wake_word = WakeWordDetector(
            WakeWordConfig(
                enabled=bool(wake_cfg.get("enabled", True)),
                backend=str(wake_cfg.get("backend", "openwakeword")),
                model_paths=tuple(wake_cfg.get("model_paths", [])),
                model_names=tuple(wake_cfg.get("model_names", [])),
                threshold=float(wake_cfg.get("threshold", 0.6)),
                patience_frames=int(wake_cfg.get("patience_frames", 2)),
                cooldown_ms=int(wake_cfg.get("cooldown_ms", 1200)),
                sample_rate=int(audio_cfg.get("sample_rate", 16000)),
                min_rms=float(wake_cfg.get("min_rms", 0.01)),
                inference_framework=str(wake_cfg.get("inference_framework", "onnx")),
                vad_threshold=float(wake_cfg.get("vad_threshold", 0.0)),
                keyword=wake_cfg.get("keyword", wake_cfg.get("agent_name", "agent")),
                keyword_aliases=tuple(wake_cfg.get("keyword_aliases", [])),
                vosk_model_path=wake_cfg.get("vosk_model_path"),
                base_path=self.config_path.parent,
                history_size=int(wake_cfg.get("history_size", 40)),
            ),
            self.bus,
        )

        self.vad = SileroVAD(
            VadConfig(
                device=str(vad_cfg.get("device", "cpu")),
                threshold=float(vad_cfg.get("threshold", 0.5)),
                min_speech_ms=int(vad_cfg.get("min_speech_ms", 300)),
                end_silence_ms=int(vad_cfg.get("end_silence_ms", 700)),
                sample_rate=int(audio_cfg.get("sample_rate", 16000)),
                min_rms=float(vad_cfg.get("min_rms", 0.01)),
                noise_floor_alpha=float(vad_cfg.get("noise_floor_alpha", 0.05)),
                noise_ratio=float(vad_cfg.get("noise_ratio", 1.5)),
                score_emit_ms=int(vad_cfg.get("score_emit_ms", 120)),
            ),
            self.bus,
        )

        backend = str(asr_cfg.get("backend", "whisper")).strip().lower()
        self.asr_backend = backend

        if backend == "vosk":
            # resolve model path
            base = self.config_path.parent
            mp = asr_cfg.get("vosk_model_path") or asr_cfg.get("model_path") or ""
            model_path = _resolve_path(base, mp)
            if model_path is None:
                # fallbacks
                candidates = [
                    (base.parent / "models" / "vosk-model-ru-0.22"),
                    (base.parent / "models" / "vosk-model-small-ru-0.22"),
                    (base / "models" / "vosk-model-ru-0.22"),
                    (base / "models" / "vosk-model-small-ru-0.22"),
                ]
                model_path = next((c for c in candidates if c.exists()), None)
            if model_path is None:
                raise FileNotFoundError(
                    "ASR backend=vosk selected, but vosk model not found. "
                    "Set asr.vosk_model_path in voice_agent/config.yaml or download a model into ./models."
                )
            self.asr = VoskASR(
                AsrVoskConfig(
                    model_path=model_path,
                    sample_rate=int(audio_cfg.get("sample_rate", 16000)),
                    max_utterance_s=int(asr_cfg.get("max_utterance_s", 10)),
                    partial_interval_ms=int(asr_cfg.get("partial_interval_ms", 220)),
                    partial_min_delta=int(asr_cfg.get("partial_min_delta", 3)),
                    min_buffer_s=float(asr_cfg.get("min_buffer_s", 0.4)),
                ),
                self.bus,
            )
        else:
            self.asr_backend = "whisper"
            self.asr = FasterWhisperASR(
                AsrConfig(
                    model=str(asr_cfg.get("model", "small")),
                    device=str(asr_cfg.get("device", "cuda")),
                    compute_type=str(asr_cfg.get("compute_type", "float16")),
                    beam_size=int(asr_cfg.get("beam_size", 1)),
                    language=str(asr_cfg.get("language", "ru")),
                    max_utterance_s=int(asr_cfg.get("max_utterance_s", 10)),
                    partial_interval_ms=int(asr_cfg.get("partial_interval_ms", 220)),
                    partial_min_delta=int(asr_cfg.get("partial_min_delta", 3)),
                    min_partial_s=float(asr_cfg.get("min_partial_s", 0.5)),
                    sample_rate=int(audio_cfg.get("sample_rate", 16000)),
                    no_speech_threshold=float(asr_cfg.get("no_speech_threshold", 0.8)),
                    log_prob_threshold=float(asr_cfg.get("log_prob_threshold", -1.0)),
                    compression_ratio_threshold=float(asr_cfg.get("compression_ratio_threshold", 2.4)),
                    min_buffer_s=float(asr_cfg.get("min_buffer_s", 0.6)),
                ),
                self.bus,
            )

        self.intent = IntentRecognizer(nlu_cfg.get("synonyms", {}))
        self.actions = ActionExecutor(self.bus)
        self.tts = TtsEngine(
            TtsConfig(
                enabled=bool(tts_cfg.get("enabled", False)),
                voice=str(tts_cfg.get("voice", "male")),
                engine=str(tts_cfg.get("engine", "piper")),
            ),
            self.bus,
        )

        # Subscriptions
        self.bus.subscribe("audio.chunk", self._on_audio)
        self.bus.subscribe("audio.level", self._on_audio_level)
        self.bus.subscribe("wake_word.scores", self._on_wake_scores)
        self.bus.subscribe("wake_word.detected", self._on_wake_word)
        self.bus.subscribe("vad.score", self._on_vad_score)
        self.bus.subscribe("vad.speech_start", self._on_vad_start)
        self.bus.subscribe("vad.speech_end", self._on_vad_end)
        self.bus.subscribe("asr.partial", self._on_partial)
        self.bus.subscribe("asr.final", self._on_final)

        self.logger.info(
            "Runtime config: wake_enabled=%s wake_backend=%s vad_device=%s asr_backend=%s sample_rate=%s",
            bool(wake_cfg.get("enabled", True)),
            wake_cfg.get("backend", "openwakeword"),
            vad_cfg.get("device", "cpu"),
            self.asr_backend,
            audio_cfg.get("sample_rate", 16000),
        )

    # --------- external controls ---------
    def set_muted(self, muted: bool) -> None:
        self.audio.set_muted(muted)
        self._status("audio.muted", {"muted": bool(muted)})

    # --------- helpers ---------
    def _status(self, kind: str, payload: dict[str, Any]) -> None:
        if self._on_status_cb:
            try:
                self._on_status_cb(kind, payload)
            except Exception:
                pass

    def _set_state(self, name: str) -> None:
        if self.state.name != name:
            self.logger.info("[STATE] %s -> %s", self.state.name, name)
            self._status("state", {"from": self.state.name, "to": name})
            self.state.name = name
            self.state.since = time.monotonic()

    # --------- event handlers ---------
    def _on_audio(self, event: Event) -> None:
        data = event.payload["data"]
        ts = float(event.payload["ts"])
        self._last_chunk = data
        self._last_chunk_ts = ts

        # Maintain preroll ring-buffer for anti-clipping of command start
        if data is not None:
            if getattr(data, "ndim", 1) > 1:
                mono = data[:, 0]
            else:
                mono = data
            if mono.dtype != np.int16:
                mono = mono.astype(np.int16, copy=False)
            if self._preroll_samples > 0:
                self._preroll_buf = np.concatenate([self._preroll_buf, mono], axis=0)
                if self._preroll_buf.size > self._preroll_samples:
                    self._preroll_buf = self._preroll_buf[-self._preroll_samples :]

        if self.state.name == "IDLE":
            self.wake_word.process_chunk(data, ts)
            return

        if self.state.name == "ARMED":
            if self._armed_since and self._armed_timeout_ms > 0:
                if (ts - self._armed_since) * 1000.0 >= self._armed_timeout_ms:
                    self.logger.info("Wake-word timed out (no speech), returning to IDLE.")
                    self._set_state("IDLE")
                    self._armed_since = None
                    self.vad.reset()
                    self.asr.reset()
                    return
            self.vad.process_chunk(data, ts)
            return

        if self.state.name == "LISTENING":
            self.vad.process_chunk(data, ts)
            # Feed ASR mono 1-D int16 to avoid shape mismatches (preroll is 1-D).
            mono_for_asr = data[:, 0] if getattr(data, "ndim", 1) > 1 else data
            if getattr(mono_for_asr, "dtype", None) != np.int16:
                mono_for_asr = mono_for_asr.astype(np.int16, copy=False)
            self.asr.accept_audio(mono_for_asr, ts)
            return

        if self.state.name == "DECODING":
            # ignore chunks until ASR emits final or timeout
            return

    def _on_audio_level(self, event: Event) -> None:
        rms = float(event.payload.get("rms", 0.0))
        peak = float(event.payload.get("peak", 0.0))
        if self._on_audio_level_cb:
            self._on_audio_level_cb(rms)
        self._status("audio.level", {"rms": rms, "peak": peak})

    def _on_wake_scores(self, event: Event) -> None:
        # Forward to UI/console
        self._status("wake.scores", dict(event.payload))

    def _on_vad_score(self, event: Event) -> None:
        self._status("vad.score", dict(event.payload))

    def _on_wake_word(self, event: Event) -> None:
        if self.state.name != "IDLE":
            return
        scores = event.payload.get("scores", {})
        best = event.payload.get("best", None)
        best_name = event.payload.get("best_name", None)
        self.logger.info("Wake-word DETECTED best=%s best_name=%s scores=%s", best, best_name, scores)
        self._status("wake.detected", {"best": best, "best_name": best_name, "scores": scores})

        self._set_state("ARMED")
        self._armed_since = float(event.payload.get("ts", time.monotonic()))
        self.vad.reset()
        self.asr.reset()

    def _on_vad_start(self, event: Event) -> None:
        # Speech started after wake-word
        if self.state.name not in {"ARMED", "LISTENING"}:
            return
        self._armed_since = None
        self.tts.stop()

        self.asr.speech_start()

        # anti-clipping: feed preroll and the last chunk (that caused speech_start)
        try:
            if self._preroll_buf.size:
                self.asr.accept_audio(self._preroll_buf.copy(), float(event.payload.get("ts", self._last_chunk_ts)))
        except Exception:
            pass
        try:
            if self._last_chunk is not None:
                lc = self._last_chunk
                lc_mono = lc[:, 0] if getattr(lc, "ndim", 1) > 1 else lc
                if getattr(lc_mono, "dtype", None) != np.int16:
                    lc_mono = lc_mono.astype(np.int16, copy=False)
                self.asr.accept_audio(lc_mono, float(event.payload.get("ts", self._last_chunk_ts)))
        except Exception:
            pass

        self._set_state("LISTENING")
        self._status("vad.speech_start", {"ts": event.payload.get("ts")})

    def _on_vad_end(self, event: Event) -> None:
        if self.state.name != "LISTENING":
            return
        self._set_state("DECODING")
        self._decode_started_at = time.monotonic()
        self._status("vad.speech_end", {"ts": event.payload.get("ts")})
        self.asr.speech_end(float(event.payload.get("ts", time.monotonic())))
        # Do NOT reset to IDLE here; wait for asr.final or timeout.

    def _on_partial(self, event: Event) -> None:
        text = str(event.payload.get("text", "")).strip()
        if not text:
            return
        if self._on_partial_cb:
            self._on_partial_cb(text)
        self._status("asr.partial", {"text": text})

    def _strip_wake_prefix(self, text: str) -> str:
        agent_name = str(self.cfg.get("wake_word", {}).get("agent_name", "")).strip().lower()
        if not agent_name:
            return text
        t = text.strip()
        low = t.lower()
        # common patterns: "бивис, ..." or "бивис ..."
        for prefix in [agent_name + ",", agent_name + ":", agent_name]:
            if low.startswith(prefix):
                return t[len(prefix):].strip(" ,:.-")
        return t

    def _on_final(self, event: Event) -> None:
        text = str(event.payload.get("text", "")).strip()
        if not text:
            # still return to IDLE
            self.vad.reset()
            self.asr.reset()
            self._set_state("IDLE")
            return

        text = self._strip_wake_prefix(text)
        normalized = self.intent.normalize(text)

        self.logger.info("ASR final: %s", text)
        self._status("asr.final", {"text": text, "normalized": normalized})

        # Deliver to callback (UI/app)
        if self._on_final_cb:
            self._on_final_cb(text)

        # Optional intent/actions
        if self._enable_actions:
            recognized = self.intent.recognize(text)
            if recognized:
                self.bus.publish(Event("agent.intent", {"name": recognized.name, "slots": recognized.slots}))
                result = self.actions.run(recognized)
                self.logger.info("Action: %s", result.message)
                if self.cfg.get("tts", {}).get("enabled", False):
                    self.tts.speak("Готово")
            else:
                self.logger.info("Intent: not understood (normalized=%s)", normalized)

        # Reset to idle cleanly
        self.vad.reset()
        self.asr.reset()
        self._decode_started_at = None
        self._set_state("IDLE")

    # --------- runtime loop ---------
    def start(self) -> None:
        if self._running.is_set():
            return
        self._running.set()
        self.audio.start()

        def loop() -> None:
            self.logger.info("Voice agent started.")
            self._status("runtime.started", {"config": str(self.config_path), "asr_backend": self.asr_backend})
            while self._running.is_set():
                # decode timeout guard
                if self.state.name == "DECODING" and self._decode_started_at is not None:
                    if (time.monotonic() - self._decode_started_at) * 1000.0 >= self._decode_timeout_ms:
                        self.logger.warning("ASR decode timeout (%dms). Returning to IDLE.", self._decode_timeout_ms)
                        self._status("asr.timeout", {"ms": self._decode_timeout_ms})
                        self.vad.reset()
                        self.asr.reset()
                        self._decode_started_at = None
                        self._set_state("IDLE")

                event = self.bus.poll(timeout=0.1)
                if event:
                    self.bus.dispatch(event)

            self.logger.info("Voice agent loop stopped.")

        self._thread = threading.Thread(target=loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        if not self._running.is_set():
            return
        self._running.clear()
        try:
            self.audio.stop()
        except Exception:
            pass
        self._status("runtime.stopped", {})
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
