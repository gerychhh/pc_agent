from __future__ import annotations

import queue
import subprocess
import threading
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk
from typing import Any

import sounddevice as sd
import yaml

from core.orchestrator import Orchestrator, sanitize_assistant_text
from core.state import get_voice_device, get_voice_engine, get_voice_model_size, set_voice_device, set_voice_engine, set_voice_model_size
from voice_agent.main import VoiceAgentRuntime


CONFIG_PATH = Path(__file__).resolve().parent / "voice_agent" / "config.yaml"


class AgentUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("PC Agent")

        self.orchestrator = Orchestrator()
        self.result_queue: queue.Queue[str] = queue.Queue()
        self.ui_queue: queue.Queue[callable] = queue.Queue()

        self.voice_config = self._load_voice_config()
        self.voice_runtime: VoiceAgentRuntime | None = None

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.chat_frame = ttk.Frame(self.notebook, padding=8)
        self.settings_frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.chat_frame, text="Chat")
        self.notebook.add(self.settings_frame, text="Voice")

        self._build_chat()
        self._build_settings()

        self.root.after(100, self._poll_results)
        self._refresh_devices(selected_index=get_voice_device())
        self._start_voice_recognition()

    # ---------------- Chat ----------------
    def _build_chat(self) -> None:
        self.chat_log = tk.Text(self.chat_frame, height=22, state=tk.DISABLED, wrap=tk.WORD)
        self.chat_log.pack(fill=tk.BOTH, expand=True)

        entry_frame = ttk.Frame(self.chat_frame)
        entry_frame.pack(fill=tk.X, pady=8)

        self.chat_entry = ttk.Entry(entry_frame)
        self.chat_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.chat_entry.bind("<Return>", lambda _: self._send_message())

        send_button = ttk.Button(entry_frame, text="Send", command=self._send_message)
        send_button.pack(side=tk.RIGHT, padx=8)

    def _append_chat(self, line: str) -> None:
        self.chat_log.configure(state=tk.NORMAL)
        self.chat_log.insert(tk.END, line + "\n")
        self.chat_log.configure(state=tk.DISABLED)
        self.chat_log.see(tk.END)

    def _send_message(self) -> None:
        text = self.chat_entry.get().strip()
        if not text:
            return
        self.chat_entry.delete(0, tk.END)
        self._run_request(text)

    def _run_request(self, text: str) -> None:
        self._append_chat(f"You: {text}")
        self._append_chat("Agent: Сейчас разберусь с задачей.")

        def worker() -> None:
            response = self.orchestrator.run(text, stateless=False, force_llm=False)
            output = sanitize_assistant_text(response) or "(no output)"
            self.result_queue.put(output)

        threading.Thread(target=worker, daemon=True).start()

    # ---------------- Voice Settings + Monitor ----------------
    def _build_settings(self) -> None:
        row = 0

        title = ttk.Label(self.settings_frame, text="Голосовой ввод: wake-word → VAD → ASR", font=("Segoe UI", 11, "bold"))
        title.grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=(0, 8))
        row += 1

        # --- ASR selection (what user asked: whisper/vosk from UI) ---
        ttk.Label(self.settings_frame, text="ASR движок").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.engine_var = tk.StringVar(value=(self.voice_config["asr"].get("backend") or get_voice_engine() or "whisper"))
        engine_box = ttk.Combobox(self.settings_frame, textvariable=self.engine_var, values=["whisper", "vosk"], state="readonly", width=12)
        engine_box.grid(row=row, column=1, sticky=tk.W, pady=2)
        row += 1

        ttk.Label(self.settings_frame, text="Whisper model").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.asr_model_var = tk.StringVar(value=str(self.voice_config["asr"].get("model", "small")))
        fw_sizes = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        ttk.Combobox(self.settings_frame, textvariable=self.asr_model_var, values=fw_sizes, state="readonly", width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1

        ttk.Label(self.settings_frame, text="Vosk model path").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.vosk_model_path_var = tk.StringVar(value=str(self.voice_config["asr"].get("vosk_model_path", "")))
        ttk.Entry(self.settings_frame, textvariable=self.vosk_model_path_var, width=44).grid(row=row, column=1, columnspan=3, sticky=tk.W)
        row += 1

        ttk.Label(self.settings_frame, text="ASR device").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.asr_device_var = tk.StringVar(value=str(self.voice_config["asr"].get("device", "cuda")))
        ttk.Combobox(self.settings_frame, textvariable=self.asr_device_var, values=["cuda", "cpu"], state="readonly", width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1

        ttk.Label(self.settings_frame, text="Compute type").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.compute_type_var = tk.StringVar(value=str(self.voice_config["asr"].get("compute_type", "float16")))
        ttk.Combobox(self.settings_frame, textvariable=self.compute_type_var, values=["float16", "int8", "int8_float16"], state="readonly", width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1

        ttk.Label(self.settings_frame, text="Max utterance (s)").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.max_utterance_var = tk.StringVar(value=str(self.voice_config["asr"].get("max_utterance_s", 10)))
        ttk.Entry(self.settings_frame, textvariable=self.max_utterance_var, width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1

        # --- Wake-word ---
        sep = ttk.Separator(self.settings_frame, orient=tk.HORIZONTAL)
        sep.grid(row=row, column=0, columnspan=4, sticky=tk.EW, pady=(8, 8))
        row += 1

        ttk.Label(self.settings_frame, text="Wake-word").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.wake_enabled_var = tk.BooleanVar(value=bool(self.voice_config["wake_word"].get("enabled", True)))
        ttk.Checkbutton(self.settings_frame, text="Enabled", variable=self.wake_enabled_var).grid(row=row, column=1, sticky=tk.W)
        row += 1

        ttk.Label(self.settings_frame, text="Wake backend").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.wake_backend_var = tk.StringVar(value=str(self.voice_config["wake_word"].get("backend", "openwakeword")))
        ttk.Combobox(self.settings_frame, textvariable=self.wake_backend_var, values=["openwakeword", "vosk"], state="readonly", width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1

        ttk.Label(self.settings_frame, text="Wake model path (.onnx)").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.wake_model_path_var = tk.StringVar(value=str((self.voice_config["wake_word"].get("model_paths") or [""])[0]))
        ttk.Entry(self.settings_frame, textvariable=self.wake_model_path_var, width=44).grid(row=row, column=1, columnspan=3, sticky=tk.W)
        row += 1

        ttk.Label(self.settings_frame, text="Wake threshold").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.wake_threshold_var = tk.StringVar(value=str(self.voice_config["wake_word"].get("threshold", 0.6)))
        ttk.Entry(self.settings_frame, textvariable=self.wake_threshold_var, width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1

        ttk.Label(self.settings_frame, text="Wake min_rms").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.wake_min_rms_var = tk.StringVar(value=str(self.voice_config["wake_word"].get("min_rms", 0.0025)))
        ttk.Entry(self.settings_frame, textvariable=self.wake_min_rms_var, width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1

        ttk.Label(self.settings_frame, text="Wake patience").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.wake_patience_var = tk.StringVar(value=str(self.voice_config["wake_word"].get("patience_frames", 2)))
        ttk.Entry(self.settings_frame, textvariable=self.wake_patience_var, width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1

        ttk.Label(self.settings_frame, text="Preroll (ms)").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.preroll_var = tk.StringVar(value=str(self.voice_config["wake_word"].get("preroll_ms", 450)))
        ttk.Entry(self.settings_frame, textvariable=self.preroll_var, width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1

        # --- Mic device ---
        sep2 = ttk.Separator(self.settings_frame, orient=tk.HORIZONTAL)
        sep2.grid(row=row, column=0, columnspan=4, sticky=tk.EW, pady=(8, 8))
        row += 1

        ttk.Label(self.settings_frame, text="Микрофон").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.device_var = tk.StringVar()
        self.device_menu = ttk.Combobox(self.settings_frame, textvariable=self.device_var, state="readonly", width=44)
        self.device_menu.grid(row=row, column=1, columnspan=2, sticky=tk.W, pady=2)
        ttk.Button(self.settings_frame, text="Refresh", command=lambda: self._refresh_devices(selected_index=get_voice_device())).grid(row=row, column=3, sticky=tk.W, padx=(6, 0))
        row += 1

        # --- VAD ---
        ttk.Label(self.settings_frame, text="VAD device").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.vad_device_var = tk.StringVar(value=str(self.voice_config["vad"].get("device", "cpu")))
        ttk.Combobox(self.settings_frame, textvariable=self.vad_device_var, values=["cpu", "cuda"], state="readonly", width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1

        ttk.Label(self.settings_frame, text="VAD threshold").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.vad_threshold_var = tk.StringVar(value=str(self.voice_config["vad"].get("threshold", 0.5)))
        ttk.Entry(self.settings_frame, textvariable=self.vad_threshold_var, width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1

        ttk.Label(self.settings_frame, text="End silence (ms)").grid(row=row, column=0, sticky=tk.W, pady=2)
        self.end_silence_var = tk.StringVar(value=str(self.voice_config["vad"].get("end_silence_ms", 700)))
        ttk.Entry(self.settings_frame, textvariable=self.end_silence_var, width=12).grid(row=row, column=1, sticky=tk.W)
        row += 1

        # --- Buttons ---
        row += 1
        ttk.Button(self.settings_frame, text="Save + Apply", command=self._save_settings).grid(row=row, column=0, sticky=tk.W, pady=(6, 6))
        ttk.Button(self.settings_frame, text="Restart voice", command=self._restart_voice_runtime).grid(row=row, column=1, sticky=tk.W, pady=(6, 6), padx=(6, 0))
        row += 1

        self.settings_status = ttk.Label(self.settings_frame, text="")
        self.settings_status.grid(row=row, column=0, columnspan=4, sticky=tk.W)
        row += 1

        # --- Monitor ---
        sep3 = ttk.Separator(self.settings_frame, orient=tk.HORIZONTAL)
        sep3.grid(row=row, column=0, columnspan=4, sticky=tk.EW, pady=(10, 10))
        row += 1

        ttk.Label(self.settings_frame, text="Live monitor", font=("Segoe UI", 10, "bold")).grid(row=row, column=0, columnspan=4, sticky=tk.W)
        row += 1

        self.pipeline_state = ttk.Label(self.settings_frame, text="State: --")
        self.pipeline_state.grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=(0, 4))
        row += 1

        self.wake_line = ttk.Label(self.settings_frame, text="Wake: --")
        self.wake_line.grid(row=row, column=0, columnspan=4, sticky=tk.W)
        row += 1

        self.wake_bar = ttk.Progressbar(self.settings_frame, orient=tk.HORIZONTAL, length=260, mode="determinate", maximum=20)
        self.wake_bar.grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=(0, 6))
        row += 1

        self.vad_line = ttk.Label(self.settings_frame, text="VAD: --")
        self.vad_line.grid(row=row, column=0, columnspan=4, sticky=tk.W)
        row += 1

        self.audio_line = ttk.Label(self.settings_frame, text="Audio RMS: --")
        self.audio_line.grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=(0, 2))
        row += 1

        self.audio_bar = ttk.Progressbar(self.settings_frame, orient=tk.HORIZONTAL, length=260, mode="determinate", maximum=100)
        self.audio_bar.grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=(0, 8))
        row += 1

        self.partial_line = ttk.Label(self.settings_frame, text="ASR partial: --", wraplength=540)
        self.partial_line.grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=(0, 2))
        row += 1

        self.final_line = ttk.Label(self.settings_frame, text="ASR final: --", wraplength=540)
        self.final_line.grid(row=row, column=0, columnspan=4, sticky=tk.W, pady=(0, 8))
        row += 1

        ttk.Label(self.settings_frame, text="Debug log").grid(row=row, column=0, columnspan=4, sticky=tk.W)
        row += 1

        log_frame = ttk.Frame(self.settings_frame)
        log_frame.grid(row=row, column=0, columnspan=4, sticky=tk.NSEW)
        self.settings_frame.rowconfigure(row, weight=1)
        self.settings_frame.columnconfigure(1, weight=1)

        self.debug_log = tk.Text(log_frame, height=10, state=tk.DISABLED, wrap=tk.NONE)
        yscroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.debug_log.yview)
        self.debug_log.configure(yscrollcommand=yscroll.set)
        self.debug_log.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        yscroll.pack(side=tk.RIGHT, fill=tk.Y)

    # ---------------- Runtime wiring ----------------
    def _poll_results(self) -> None:
        while True:
            try:
                result = self.result_queue.get_nowait()
            except queue.Empty:
                break
            self._append_chat(f"Agent: {result}")
            self._append_chat("Agent: Готово")
        while True:
            try:
                task = self.ui_queue.get_nowait()
            except queue.Empty:
                break
            task()
        self.root.after(100, self._poll_results)

    def _restart_voice_runtime(self) -> None:
        try:
            if self.voice_runtime:
                self.voice_runtime.stop()
        except Exception:
            pass
        self.voice_runtime = None
        self._start_voice_recognition()
        self._set_label_safe(self.settings_status, "Voice runtime restarted.")

    def _start_voice_recognition(self) -> None:
        if self.voice_runtime:
            return
        self._set_label_safe(self.settings_status, "Starting voice runtime...")

        def worker() -> None:
            try:
                def on_final(text: str) -> None:
                    cleaned = text.strip()
                    if not cleaned:
                        return
                    self.ui_queue.put(lambda: self._append_chat(f"Voice: {cleaned}"))
                    self.ui_queue.put(lambda: self._run_request(cleaned))

                def on_partial(text: str) -> None:
                    self.ui_queue.put(lambda: self.partial_line.configure(text=f"ASR partial: {text}"))

                def on_audio_level(rms: float) -> None:
                    # RMS normalized 0..1; scale for progressbar
                    level = min(100, int(rms * 5000))
                    self.ui_queue.put(lambda: self._update_audio_level(level, rms))

                def on_status(kind: str, payload: dict[str, Any]) -> None:
                    self.ui_queue.put(lambda: self._handle_status(kind, payload))

                runtime = VoiceAgentRuntime(
                    CONFIG_PATH,
                    on_final=on_final,
                    on_partial=on_partial,
                    on_audio_level=on_audio_level,
                    on_status=on_status,
                    enable_actions=False,
                )
                runtime.start()
                self.voice_runtime = runtime
                self._set_label_safe(self.settings_status, "Voice runtime started. See console logs + monitor below.")
            except Exception as exc:
                self._set_label_safe(self.settings_status, f"Voice runtime error: {exc}")

        threading.Thread(target=worker, daemon=True).start()

    def _handle_status(self, kind: str, payload: dict[str, Any]) -> None:
        # Minimal, clear UI updates
        now = time.strftime("%H:%M:%S")
        if kind == "state":
            self.pipeline_state.configure(text=f"State: {payload.get('to')}")
        elif kind == "wake.scores":
            best = float(payload.get("best", 0.0))
            timeline = payload.get("timeline", "")
            best_name = payload.get("best_name", "")
            self.wake_line.configure(text=f"Wake[{best_name}]: {best:.2f}  [{timeline}]")
            self.wake_bar["value"] = int(payload.get("bar", 0))
        elif kind == "wake.detected":
            best = payload.get("best")
            best_name = payload.get("best_name")
            self._append_debug(f"[{now}] WAKE DETECTED: {best_name} score={best}")
        elif kind == "vad.score":
            prob = float(payload.get("prob", 0.0))
            speaking = bool(payload.get("speaking", False))
            self.vad_line.configure(text=f"VAD: prob={prob:.2f} speaking={'YES' if speaking else 'no'}")
        elif kind == "vad.speech_start":
            self._append_debug(f"[{now}] VAD speech_start")
        elif kind == "vad.speech_end":
            self._append_debug(f"[{now}] VAD speech_end")
        elif kind == "asr.partial":
            self.partial_line.configure(text=f"ASR partial: {payload.get('text','')}")
        elif kind == "asr.final":
            self.final_line.configure(text=f"ASR final: {payload.get('text','')}")
            self._append_debug(f"[{now}] ASR FINAL: {payload.get('text','')}")
        elif kind in {"runtime.started", "runtime.stopped", "asr.timeout", "audio.muted"}:
            self._append_debug(f"[{now}] {kind}: {payload}")
        # Keep log readable: only important kinds are printed here.

    def _append_debug(self, line: str) -> None:
        self.debug_log.configure(state=tk.NORMAL)
        self.debug_log.insert(tk.END, line + "\n")
        self.debug_log.configure(state=tk.DISABLED)
        self.debug_log.see(tk.END)

    def _update_audio_level(self, level: int, rms: float) -> None:
        self.audio_bar["value"] = level
        self.audio_line.configure(text=f"Audio RMS: {rms:.4f}")

    # ---------------- Save/Load config ----------------
    def _load_voice_config(self) -> dict[str, dict[str, Any]]:
        if CONFIG_PATH.exists():
            data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        else:
            data = {}

        # Ensure all sections exist
        return {
            "audio": dict(data.get("audio", {})),
            "vad": dict(data.get("vad", {})),
            "asr": dict(data.get("asr", {})),
            "wake_word": dict(data.get("wake_word", {})),
            "voice": dict(data.get("voice", {})),
            "logging": dict(data.get("logging", {})),
        }

    def _save_voice_config(self, config: dict[str, dict[str, Any]]) -> None:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if CONFIG_PATH.exists():
            existing = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        existing.update(config)
        CONFIG_PATH.write_text(yaml.safe_dump(existing, sort_keys=False, allow_unicode=True), encoding="utf-8")

    def _save_settings(self) -> None:
        # UI -> config.yaml + core.state for convenience
        engine = (self.engine_var.get() or "whisper").strip().lower()
        if engine:
            set_voice_engine(engine)

        device_index = self._selected_device_index()
        set_voice_device(device_index)

        # Keep legacy "model size" value for other parts of app
        model_size = (self.asr_model_var.get() or "small").strip().lower()
        set_voice_model_size(model_size)

        # audio + vad
        self.voice_config["audio"]["device"] = device_index
        self.voice_config["vad"]["device"] = (self.vad_device_var.get() or "cpu").strip().lower()
        self.voice_config["vad"]["threshold"] = float(self.vad_threshold_var.get() or 0.5)
        self.voice_config["vad"]["end_silence_ms"] = int(self.end_silence_var.get() or 700)

        # asr
        self.voice_config["asr"]["backend"] = engine
        self.voice_config["asr"]["model"] = model_size
        self.voice_config["asr"]["device"] = (self.asr_device_var.get() or "cuda").strip().lower()
        self.voice_config["asr"]["compute_type"] = (self.compute_type_var.get() or "float16").strip()
        self.voice_config["asr"]["max_utterance_s"] = int(self.max_utterance_var.get() or 10)
        self.voice_config["asr"]["vosk_model_path"] = (self.vosk_model_path_var.get() or "").strip()

        # wake word
        self.voice_config["wake_word"]["enabled"] = bool(self.wake_enabled_var.get())
        self.voice_config["wake_word"]["backend"] = (self.wake_backend_var.get() or "openwakeword").strip().lower()
        self.voice_config["wake_word"]["model_paths"] = [self.wake_model_path_var.get().strip()] if self.wake_model_path_var.get().strip() else []
        self.voice_config["wake_word"]["threshold"] = float(self.wake_threshold_var.get() or 0.6)
        self.voice_config["wake_word"]["min_rms"] = float(self.wake_min_rms_var.get() or 0.0025)
        self.voice_config["wake_word"]["patience_frames"] = int(self.wake_patience_var.get() or 2)
        self.voice_config["wake_word"]["preroll_ms"] = int(self.preroll_var.get() or 450)

        # logging defaults for clarity
        self.voice_config["logging"]["level"] = self.voice_config["logging"].get("level", "info")

        self._save_voice_config(self.voice_config)
        self._set_label_safe(self.settings_status, "Saved. Restarting voice runtime to apply...")
        self._restart_voice_runtime()

    # ---------------- devices ----------------
    def _refresh_devices(self, selected_index: int | None = None) -> None:
        devices = []
        try:
            for idx, device in enumerate(sd.query_devices()):
                if device.get("max_input_channels", 0) > 0:
                    name = str(device.get("name") or f"Device {idx}")
                    devices.append((idx, f"{idx}: {name}"))
        except Exception:
            devices = []
        self.device_options = devices
        display_values = [label for _, label in devices]
        self.device_menu.configure(values=display_values)
        if selected_index is not None:
            for idx, label in devices:
                if idx == selected_index:
                    self.device_var.set(label)
                    break
            else:
                self.device_var.set(display_values[0] if display_values else "")
        else:
            self.device_var.set(display_values[0] if display_values else "")

    def _selected_device_index(self) -> int | None:
        current = self.device_var.get().strip()
        if not current:
            return None
        try:
            return int(current.split(":", 1)[0])
        except ValueError:
            return None

    # ---------------- UI helpers ----------------
    def _set_label_safe(self, label: ttk.Label, text: str) -> None:
        def apply() -> None:
            label.configure(text=text)

        if threading.current_thread() is threading.main_thread():
            apply()
        else:
            self.ui_queue.put(apply)

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    AgentUI().run()


if __name__ == "__main__":
    main()
