from __future__ import annotations

import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import yaml
import sounddevice as sd

from core.orchestrator import Orchestrator, sanitize_assistant_text
from core.voice import VoiceInput
from core.state import (
    get_voice_device,
    get_voice_engine,
    get_voice_model_size,
    set_voice_device,
    set_voice_engine,
    set_voice_model_size,
)
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
        self.notebook.add(self.settings_frame, text="Settings")

        self._build_chat()
        self._build_settings()

        self.root.after(100, self._poll_results)
        self._load_voice_model()
        self._start_voice_recognition()
        if self.startup_voice_var.get():
            self._test_voice()

    def _build_chat(self) -> None:
        self.chat_log = tk.Text(self.chat_frame, height=20, state=tk.DISABLED, wrap=tk.WORD)
        self.chat_log.pack(fill=tk.BOTH, expand=True)

        entry_frame = ttk.Frame(self.chat_frame)
        entry_frame.pack(fill=tk.X, pady=8)

        self.chat_entry = ttk.Entry(entry_frame)
        self.chat_entry.pack(side=tk.LEFT, fill=tk.X, expand=True)
        self.chat_entry.bind("<Return>", lambda _: self._send_message())

        send_button = ttk.Button(entry_frame, text="Send", command=self._send_message)
        send_button.pack(side=tk.RIGHT, padx=8)

    def _build_settings(self) -> None:
        settings_title = ttk.Label(self.settings_frame, text="Настройки голосового ввода")
        settings_title.grid(row=0, column=0, columnspan=2, sticky=tk.W, pady=(0, 8))

        settings_note = ttk.Label(
            self.settings_frame,
            text=(
                "Описание: эти настройки помогают не обрезать длинные фразы. "
                "Рекомендуем: конец тишины 700–900 мс и максимум фразы 12–15 с."
            ),
            wraplength=420,
        )
        settings_note.grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(0, 8))

        name_label = ttk.Label(self.settings_frame, text="Имя агента")
        name_label.grid(row=2, column=0, sticky=tk.W, pady=4)
        self.agent_name_var = tk.StringVar(value=str(self.voice_config["voice"].get("agent_name", "Агент")))
        name_entry = ttk.Entry(self.settings_frame, textvariable=self.agent_name_var, width=18)
        name_entry.grid(row=2, column=1, sticky=tk.W, pady=4)

        voice_name_label = ttk.Label(self.settings_frame, text="Имя голоса TTS")
        voice_name_label.grid(row=3, column=0, sticky=tk.W, pady=4)
        self.voice_name_var = tk.StringVar(value=str(self.voice_config["voice"].get("voice_name", "Microsoft Dmitry")))
        voice_name_entry = ttk.Entry(self.settings_frame, textvariable=self.voice_name_var, width=18)
        voice_name_entry.grid(row=3, column=1, sticky=tk.W, pady=4)

        self.startup_voice_var = tk.BooleanVar(value=bool(self.voice_config["voice"].get("startup_voice", True)))
        startup_voice = ttk.Checkbutton(
            self.settings_frame,
            text="Озвучить при запуске",
            variable=self.startup_voice_var,
        )
        startup_voice.grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=4)

        engine_label = ttk.Label(self.settings_frame, text="Движок распознавания")
        engine_label.grid(row=5, column=0, sticky=tk.W, pady=4)
        self.engine_var = tk.StringVar(value=get_voice_engine() or "whisper")
        engine_box = ttk.Combobox(
            self.settings_frame,
            textvariable=self.engine_var,
            values=["vosk", "whisper"],
            state="readonly",
            width=12,
        )
        engine_box.grid(row=5, column=1, sticky=tk.W, pady=4)

        size_label = ttk.Label(self.settings_frame, text="Размер модели")
        size_label.grid(row=6, column=0, sticky=tk.W, pady=4)
        self.model_size_var = tk.StringVar(value=get_voice_model_size() or "small")
        whisper_sizes = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        size_box = ttk.Combobox(
            self.settings_frame,
            textvariable=self.model_size_var,
            values=whisper_sizes,
            state="readonly",
            width=12,
        )
        size_box.grid(row=6, column=1, sticky=tk.W, pady=4)

        device_label = ttk.Label(self.settings_frame, text="Индекс микрофона")
        device_label.grid(row=7, column=0, sticky=tk.W, pady=4)
        self.device_options: list[tuple[int, str]] = []
        device_value = get_voice_device()
        self.device_var = tk.StringVar()
        self.device_menu = ttk.Combobox(
            self.settings_frame,
            textvariable=self.device_var,
            state="readonly",
            width=28,
        )
        self.device_menu.grid(row=7, column=1, sticky=tk.W, pady=4)
        refresh_button = ttk.Button(self.settings_frame, text="Обновить", command=self._refresh_devices)
        refresh_button.grid(row=7, column=2, sticky=tk.W, padx=(6, 0))
        self._refresh_devices(selected_index=device_value)

        audio_title = ttk.Label(self.settings_frame, text="Захват аудио")
        audio_title.grid(row=8, column=0, columnspan=2, sticky=tk.W, pady=(12, 4))

        self.sample_rate_var = tk.StringVar(value=str(self.voice_config["audio"].get("sample_rate", 16000)))
        ttk.Label(self.settings_frame, text="Частота (Гц)").grid(row=9, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.settings_frame, textvariable=self.sample_rate_var, width=12).grid(row=9, column=1, sticky=tk.W)

        self.chunk_ms_var = tk.StringVar(value=str(self.voice_config["audio"].get("chunk_ms", 20)))
        ttk.Label(self.settings_frame, text="Чанк (мс)").grid(row=10, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.settings_frame, textvariable=self.chunk_ms_var, width=12).grid(row=10, column=1, sticky=tk.W)

        vad_title = ttk.Label(self.settings_frame, text="VAD (детекция речи)")
        vad_title.grid(row=11, column=0, columnspan=2, sticky=tk.W, pady=(12, 4))

        # Move VAD to GPU when possible (silero_vad supports CUDA via torch)
        self.vad_device_var = tk.StringVar(value=str(self.voice_config["vad"].get("device", "cpu")))
        ttk.Label(self.settings_frame, text="VAD устройство").grid(row=12, column=0, sticky=tk.W, pady=2)
        ttk.Combobox(
            self.settings_frame,
            textvariable=self.vad_device_var,
            values=["cpu", "cuda"],
            state="readonly",
            width=12,
        ).grid(row=12, column=1, sticky=tk.W)

        self.vad_threshold_var = tk.StringVar(value=str(self.voice_config["vad"].get("threshold", 0.5)))
        ttk.Label(self.settings_frame, text="Порог").grid(row=13, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.settings_frame, textvariable=self.vad_threshold_var, width=12).grid(row=13, column=1, sticky=tk.W)

        self.min_speech_var = tk.StringVar(value=str(self.voice_config["vad"].get("min_speech_ms", 200)))
        ttk.Label(self.settings_frame, text="Мин. речь (мс)").grid(row=14, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.settings_frame, textvariable=self.min_speech_var, width=12).grid(row=14, column=1, sticky=tk.W)

        self.end_silence_var = tk.StringVar(value=str(self.voice_config["vad"].get("end_silence_ms", 500)))
        ttk.Label(self.settings_frame, text="Конец тишины (мс)").grid(row=15, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.settings_frame, textvariable=self.end_silence_var, width=12).grid(row=15, column=1, sticky=tk.W)

        self.vad_min_rms_var = tk.StringVar(value=str(self.voice_config["vad"].get("min_rms", 0.01)))
        ttk.Label(self.settings_frame, text="Мин. громкость (RMS)").grid(row=16, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.settings_frame, textvariable=self.vad_min_rms_var, width=12).grid(row=16, column=1, sticky=tk.W)

        self.vad_noise_ratio_var = tk.StringVar(value=str(self.voice_config["vad"].get("noise_ratio", 1.5)))
        ttk.Label(self.settings_frame, text="Шумовой множитель").grid(row=17, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.settings_frame, textvariable=self.vad_noise_ratio_var, width=12).grid(row=17, column=1, sticky=tk.W)

        self.vad_noise_alpha_var = tk.StringVar(value=str(self.voice_config["vad"].get("noise_floor_alpha", 0.05)))
        ttk.Label(self.settings_frame, text="Скорость оценки шума").grid(row=18, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.settings_frame, textvariable=self.vad_noise_alpha_var, width=12).grid(row=18, column=1, sticky=tk.W)

        asr_title = ttk.Label(self.settings_frame, text="ASR (faster-whisper)")
        asr_title.grid(row=19, column=0, columnspan=2, sticky=tk.W, pady=(12, 4))

        self.asr_model_var = tk.StringVar(value=str(self.voice_config["asr"].get("model", "small")))
        ttk.Label(self.settings_frame, text="Модель").grid(row=20, column=0, sticky=tk.W, pady=2)
        fw_sizes = ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3"]
        ttk.Combobox(
            self.settings_frame,
            textvariable=self.asr_model_var,
            values=fw_sizes,
            state="readonly",
            width=12,
        ).grid(row=20, column=1, sticky=tk.W)

        self.asr_device_var = tk.StringVar(value=str(self.voice_config["asr"].get("device", "cuda")))
        ttk.Label(self.settings_frame, text="Устройство").grid(row=21, column=0, sticky=tk.W, pady=2)
        ttk.Combobox(
            self.settings_frame,
            textvariable=self.asr_device_var,
            values=["cuda", "cpu"],
            state="readonly",
            width=12,
        ).grid(row=21, column=1, sticky=tk.W)

        self.compute_type_var = tk.StringVar(value=str(self.voice_config["asr"].get("compute_type", "float16")))
        ttk.Label(self.settings_frame, text="Тип вычислений").grid(row=22, column=0, sticky=tk.W, pady=2)
        ttk.Combobox(
            self.settings_frame,
            textvariable=self.compute_type_var,
            values=["float16", "int8", "int8_float16"],
            state="readonly",
            width=12,
        ).grid(row=22, column=1, sticky=tk.W)

        self.beam_size_var = tk.StringVar(value=str(self.voice_config["asr"].get("beam_size", 2)))
        ttk.Label(self.settings_frame, text="Beam size").grid(row=23, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.settings_frame, textvariable=self.beam_size_var, width=12).grid(row=23, column=1, sticky=tk.W)

        self.max_utterance_var = tk.StringVar(value=str(self.voice_config["asr"].get("max_utterance_s", 10)))
        ttk.Label(self.settings_frame, text="Макс. фраза (с)").grid(row=24, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.settings_frame, textvariable=self.max_utterance_var, width=12).grid(row=24, column=1, sticky=tk.W)

        self.partial_interval_var = tk.StringVar(value=str(self.voice_config["asr"].get("partial_interval_ms", 150)))
        ttk.Label(self.settings_frame, text="Интервал partial (мс)").grid(row=25, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.settings_frame, textvariable=self.partial_interval_var, width=12).grid(row=25, column=1, sticky=tk.W)

        self.no_speech_threshold_var = tk.StringVar(value=str(self.voice_config["asr"].get("no_speech_threshold", 0.8)))
        ttk.Label(self.settings_frame, text="No-speech threshold").grid(row=26, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.settings_frame, textvariable=self.no_speech_threshold_var, width=12).grid(row=26, column=1, sticky=tk.W)

        self.log_prob_threshold_var = tk.StringVar(value=str(self.voice_config["asr"].get("log_prob_threshold", -1.0)))
        ttk.Label(self.settings_frame, text="Log prob threshold").grid(row=27, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.settings_frame, textvariable=self.log_prob_threshold_var, width=12).grid(row=27, column=1, sticky=tk.W)

        self.voice_status = ttk.Label(self.settings_frame, text="Модель голоса: не загружена")
        self.voice_status.grid(row=28, column=0, columnspan=2, sticky=tk.W, pady=(8, 2))

        self.recognition_status = ttk.Label(self.settings_frame, text="Распознавание: выключено")
        self.recognition_status.grid(row=29, column=0, columnspan=2, sticky=tk.W, pady=(0, 8))

        self.noise_label = ttk.Label(self.settings_frame, text="Индикатор шума: --")
        self.noise_label.grid(row=30, column=0, columnspan=2, sticky=tk.W, pady=(0, 4))
        self.noise_bar = ttk.Progressbar(
            self.settings_frame,
            orient=tk.HORIZONTAL,
            length=180,
            mode="determinate",
            maximum=100,
        )
        self.noise_bar.grid(row=31, column=0, columnspan=2, sticky=tk.W, pady=(0, 8))

        load_button = ttk.Button(self.settings_frame, text="Загрузить модель", command=self._load_voice_model)
        load_button.grid(row=32, column=0, columnspan=2, sticky=tk.W, pady=(0, 8))

        test_voice_button = ttk.Button(self.settings_frame, text="Проверить голос", command=self._test_voice)
        test_voice_button.grid(row=33, column=0, columnspan=2, sticky=tk.W, pady=(0, 8))

        save_button = ttk.Button(self.settings_frame, text="Сохранить настройки", command=self._save_settings)
        save_button.grid(row=34, column=0, columnspan=2, sticky=tk.W, pady=8)

        self.settings_status = ttk.Label(self.settings_frame, text="")
        self.settings_status.grid(row=35, column=0, columnspan=2, sticky=tk.W)

        self.settings_frame.columnconfigure(1, weight=1)

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

    def _save_settings(self) -> None:
        # Save + APPLY: if model/device changed, we restart voice runtime
        prev_engine = get_voice_engine() or "whisper"
        prev_model_size = get_voice_model_size() or "small"
        prev_device_index = get_voice_device()
        prev_asr = dict(self.voice_config.get("asr", {}))
        prev_audio = dict(self.voice_config.get("audio", {}))
        prev_vad = dict(self.voice_config.get("vad", {}))

        engine = self.engine_var.get().strip().lower()
        model_size = self.model_size_var.get().strip().lower()
        device_index = self._selected_device_index()

        if engine:
            set_voice_engine(engine)
        if model_size:
            set_voice_model_size(model_size)
        set_voice_device(device_index)

        self.voice_config["audio"]["sample_rate"] = int(self.sample_rate_var.get() or 16000)
        self.voice_config["audio"]["chunk_ms"] = int(self.chunk_ms_var.get() or 20)
        self.voice_config["audio"]["device"] = device_index
        self.voice_config["vad"]["device"] = (self.vad_device_var.get() or "cpu").strip().lower()
        self.voice_config["vad"]["threshold"] = float(self.vad_threshold_var.get() or 0.5)
        self.voice_config["vad"]["min_speech_ms"] = int(self.min_speech_var.get() or 200)
        self.voice_config["vad"]["end_silence_ms"] = int(self.end_silence_var.get() or 500)
        self.voice_config["vad"]["min_rms"] = float(self.vad_min_rms_var.get() or 0.01)
        self.voice_config["vad"]["noise_ratio"] = float(self.vad_noise_ratio_var.get() or 1.5)
        self.voice_config["vad"]["noise_floor_alpha"] = float(self.vad_noise_alpha_var.get() or 0.05)
        self.voice_config["asr"]["model"] = self.asr_model_var.get() or "small"
        asr_device = (self.asr_device_var.get() or "cuda").strip().lower()
        self.voice_config["asr"]["device"] = asr_device
        # sensible defaults: GPU -> float16, CPU -> int8
        compute_type = (self.compute_type_var.get() or ("float16" if asr_device == "cuda" else "int8")).strip()
        self.voice_config["asr"]["compute_type"] = compute_type
        self.voice_config["asr"]["beam_size"] = int(self.beam_size_var.get() or 2)
        self.voice_config["asr"]["max_utterance_s"] = int(self.max_utterance_var.get() or 10)
        self.voice_config["asr"]["partial_interval_ms"] = int(self.partial_interval_var.get() or 150)
        self.voice_config["asr"]["no_speech_threshold"] = float(self.no_speech_threshold_var.get() or 0.8)
        self.voice_config["asr"]["log_prob_threshold"] = float(self.log_prob_threshold_var.get() or -1.0)
        self.voice_config["voice"]["agent_name"] = self.agent_name_var.get() or "Агент"
        self.voice_config["voice"]["voice_name"] = self.voice_name_var.get() or "Microsoft Dmitry"
        self.voice_config["voice"]["startup_voice"] = bool(self.startup_voice_var.get())
        self._save_voice_config(self.voice_config)

        # Apply changes immediately
        needs_reload_core = (engine != prev_engine) or (model_size != prev_model_size) or (device_index != prev_device_index)

        new_asr = dict(self.voice_config.get("asr", {}))
        new_audio = dict(self.voice_config.get("audio", {}))
        new_vad = dict(self.voice_config.get("vad", {}))
        needs_restart_runtime = (new_asr != prev_asr) or (new_audio != prev_audio) or (new_vad != prev_vad)

        if needs_reload_core:
            self._load_voice_model()

        if needs_restart_runtime:
            self._restart_voice_runtime()

        self.settings_status.configure(text="Настройки сохранены и применены.")

    def _restart_voice_runtime(self) -> None:
        # stop -> start with fresh config
        try:
            if self.voice_runtime:
                self.voice_runtime.stop()
                self.voice_runtime = None
        except Exception:
            self.voice_runtime = None

        # start again
        self._start_voice_recognition()

    def _load_voice_config(self) -> dict[str, dict[str, object]]:
        if CONFIG_PATH.exists():
            data = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        else:
            data = {}
        return {
            "audio": data.get("audio", {}),
            "vad": data.get("vad", {}),
            "asr": data.get("asr", {}),
            "voice": data.get("voice", {}),
        }

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
                self.device_var.set("")
        elif display_values:
            self.device_var.set(display_values[0])
        else:
            self.device_var.set("")

    def _selected_device_index(self) -> int | None:
        current = self.device_var.get().strip()
        if not current:
            return None
        try:
            return int(current.split(":", 1)[0])
        except ValueError:
            return None

    def _save_voice_config(self, config: dict[str, dict[str, object]]) -> None:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if CONFIG_PATH.exists():
            existing = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        existing.update(config)
        CONFIG_PATH.write_text(yaml.safe_dump(existing, sort_keys=False, allow_unicode=True), encoding="utf-8")

    def _load_voice_model(self) -> None:
        self._set_label_safe(self.voice_status, "Модель голоса: загружается...")
        engine = self.engine_var.get().strip().lower()
        model_size = self.model_size_var.get().strip().lower()
        device = get_voice_device()
        display_engine = engine or "vosk"
        display_size = model_size or "default"

        def worker() -> None:
            try:
                VoiceInput(
                    device=device,
                    engine=engine,
                    model_size=model_size,
                )
            except Exception as exc:
                self._set_label_safe(self.voice_status, f"Модель голоса: ошибка ({exc})")
            else:
                self._set_label_safe(
                    self.voice_status,
                    f"Модель голоса: загружена ({display_engine}, {display_size})",
                )

        threading.Thread(target=worker, daemon=True).start()

    def _test_voice(self) -> None:
        voice_name = self.voice_name_var.get().strip() or "Microsoft Dmitry"
        text = "Голос активен."
        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"try {{ $synth.SelectVoice('{voice_name}') }} catch {{}}; "
            "$synth.Rate = 2; "
            "$synth.Volume = 100; "
            f"$synth.Speak('{text}')"
        )
        subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            check=False,
        )

    def _start_voice_recognition(self) -> None:
        if self.voice_runtime:
            return
        self._set_label_safe(self.recognition_status, "Распознавание: запускается...")

        def worker() -> None:
            try:
                def on_final(text: str) -> None:
                    cleaned = text.strip()
                    if not cleaned:
                        return
                    self.ui_queue.put(lambda: self._run_request(cleaned))

                def on_audio_level(rms: float) -> None:
                    level = min(100, int(rms * 2))
                    self.ui_queue.put(lambda: self._update_noise_level(level, rms))

                runtime = VoiceAgentRuntime(
                    CONFIG_PATH,
                    on_final=on_final,
                    on_audio_level=on_audio_level,
                    enable_actions=False,
                )
                runtime.start()
                self.voice_runtime = runtime
                asr_cfg = runtime.cfg.get("asr", {})
                model = asr_cfg.get("model", "unknown")
                device = asr_cfg.get("device", "cpu")
                compute_type = asr_cfg.get("compute_type", "")
                extra = f"{model} ({device}{'/' + compute_type if compute_type else ''})"
                self._set_label_safe(self.recognition_status, f"Распознавание: активно ({extra})")
            except Exception as exc:
                self._set_label_safe(self.recognition_status, f"Распознавание: ошибка ({exc})")

        threading.Thread(target=worker, daemon=True).start()

    def _update_noise_level(self, level: int, rms: float) -> None:
        self.noise_bar["value"] = level
        self.noise_label.configure(text=f"Индикатор шума: {rms:.1f}")

    def run(self) -> None:
        self.root.mainloop()

    def _set_label_safe(self, label: ttk.Label, text: str) -> None:
        def apply() -> None:
            label.configure(text=text)

        if threading.current_thread() is threading.main_thread():
            apply()
        else:
            self.ui_queue.put(apply)


def main() -> None:
    AgentUI().run()


if __name__ == "__main__":
    main()

# --- Safety: ensure AgentUI has _set_label_safe even if a patch removed it ---
def _agentui_set_label_safe(self, widget, text: str) -> None:
    try:
        widget.after(0, lambda: widget.configure(text=text))
    except Exception:
        try:
            widget.configure(text=text)
        except Exception:
            pass

try:
    if "AgentUI" in globals() and not hasattr(AgentUI, "_set_label_safe"):
        AgentUI._set_label_safe = _agentui_set_label_safe  # type: ignore[attr-defined]
except Exception:
    pass
