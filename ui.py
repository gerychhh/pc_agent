from __future__ import annotations

import queue
import subprocess
import threading
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import yaml

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


CONFIG_PATH = Path(__file__).resolve().parent / "voice_agent" / "config.yaml"


class AgentUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("PC Agent")
        self.orchestrator = Orchestrator()
        self.result_queue: queue.Queue[str] = queue.Queue()
        self.voice_config = self._load_voice_config()

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
        size_entry = ttk.Entry(self.settings_frame, textvariable=self.model_size_var, width=12)
        size_entry.grid(row=6, column=1, sticky=tk.W, pady=4)

        device_label = ttk.Label(self.settings_frame, text="Индекс микрофона")
        device_label.grid(row=7, column=0, sticky=tk.W, pady=4)
        device_value = get_voice_device()
        self.device_var = tk.StringVar(value="" if device_value is None else str(device_value))
        device_entry = ttk.Entry(self.settings_frame, textvariable=self.device_var, width=12)
        device_entry.grid(row=7, column=1, sticky=tk.W, pady=4)

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

        self.vad_threshold_var = tk.StringVar(value=str(self.voice_config["vad"].get("threshold", 0.5)))
        ttk.Label(self.settings_frame, text="Порог").grid(row=12, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.settings_frame, textvariable=self.vad_threshold_var, width=12).grid(row=12, column=1, sticky=tk.W)

        self.min_speech_var = tk.StringVar(value=str(self.voice_config["vad"].get("min_speech_ms", 200)))
        ttk.Label(self.settings_frame, text="Мин. речь (мс)").grid(row=13, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.settings_frame, textvariable=self.min_speech_var, width=12).grid(row=13, column=1, sticky=tk.W)

        self.end_silence_var = tk.StringVar(value=str(self.voice_config["vad"].get("end_silence_ms", 500)))
        ttk.Label(self.settings_frame, text="Конец тишины (мс)").grid(row=14, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.settings_frame, textvariable=self.end_silence_var, width=12).grid(row=14, column=1, sticky=tk.W)

        asr_title = ttk.Label(self.settings_frame, text="ASR (faster-whisper)")
        asr_title.grid(row=15, column=0, columnspan=2, sticky=tk.W, pady=(12, 4))

        self.asr_model_var = tk.StringVar(value=str(self.voice_config["asr"].get("model", "small")))
        ttk.Label(self.settings_frame, text="Модель").grid(row=16, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.settings_frame, textvariable=self.asr_model_var, width=12).grid(row=16, column=1, sticky=tk.W)

        self.asr_device_var = tk.StringVar(value=str(self.voice_config["asr"].get("device", "cuda")))
        ttk.Label(self.settings_frame, text="Устройство").grid(row=17, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.settings_frame, textvariable=self.asr_device_var, width=12).grid(row=17, column=1, sticky=tk.W)

        self.compute_type_var = tk.StringVar(value=str(self.voice_config["asr"].get("compute_type", "float16")))
        ttk.Label(self.settings_frame, text="Тип вычислений").grid(row=18, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.settings_frame, textvariable=self.compute_type_var, width=12).grid(row=18, column=1, sticky=tk.W)

        self.beam_size_var = tk.StringVar(value=str(self.voice_config["asr"].get("beam_size", 2)))
        ttk.Label(self.settings_frame, text="Beam size").grid(row=19, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.settings_frame, textvariable=self.beam_size_var, width=12).grid(row=19, column=1, sticky=tk.W)

        self.max_utterance_var = tk.StringVar(value=str(self.voice_config["asr"].get("max_utterance_s", 10)))
        ttk.Label(self.settings_frame, text="Макс. фраза (с)").grid(row=20, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.settings_frame, textvariable=self.max_utterance_var, width=12).grid(row=20, column=1, sticky=tk.W)

        self.partial_interval_var = tk.StringVar(value=str(self.voice_config["asr"].get("partial_interval_ms", 150)))
        ttk.Label(self.settings_frame, text="Интервал partial (мс)").grid(row=21, column=0, sticky=tk.W, pady=2)
        ttk.Entry(self.settings_frame, textvariable=self.partial_interval_var, width=12).grid(row=21, column=1, sticky=tk.W)

        self.voice_status = ttk.Label(self.settings_frame, text="Модель голоса: не загружена")
        self.voice_status.grid(row=22, column=0, columnspan=2, sticky=tk.W, pady=(8, 2))

        load_button = ttk.Button(self.settings_frame, text="Загрузить модель", command=self._load_voice_model)
        load_button.grid(row=23, column=0, columnspan=2, sticky=tk.W, pady=(0, 8))

        test_voice_button = ttk.Button(self.settings_frame, text="Проверить голос", command=self._test_voice)
        test_voice_button.grid(row=24, column=0, columnspan=2, sticky=tk.W, pady=(0, 8))

        save_button = ttk.Button(self.settings_frame, text="Сохранить настройки", command=self._save_settings)
        save_button.grid(row=25, column=0, columnspan=2, sticky=tk.W, pady=8)

        self.settings_status = ttk.Label(self.settings_frame, text="")
        self.settings_status.grid(row=26, column=0, columnspan=2, sticky=tk.W)

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
        self.root.after(100, self._poll_results)

    def _save_settings(self) -> None:
        engine = self.engine_var.get().strip().lower()
        model_size = self.model_size_var.get().strip().lower()
        device_text = self.device_var.get().strip()
        device_index = int(device_text) if device_text else None

        if engine:
            set_voice_engine(engine)
        if model_size:
            set_voice_model_size(model_size)
        set_voice_device(device_index)

        self.voice_config["audio"]["sample_rate"] = int(self.sample_rate_var.get() or 16000)
        self.voice_config["audio"]["chunk_ms"] = int(self.chunk_ms_var.get() or 20)
        self.voice_config["vad"]["threshold"] = float(self.vad_threshold_var.get() or 0.5)
        self.voice_config["vad"]["min_speech_ms"] = int(self.min_speech_var.get() or 200)
        self.voice_config["vad"]["end_silence_ms"] = int(self.end_silence_var.get() or 500)
        self.voice_config["asr"]["model"] = self.asr_model_var.get() or "small"
        self.voice_config["asr"]["device"] = self.asr_device_var.get() or "cuda"
        self.voice_config["asr"]["compute_type"] = self.compute_type_var.get() or "float16"
        self.voice_config["asr"]["beam_size"] = int(self.beam_size_var.get() or 2)
        self.voice_config["asr"]["max_utterance_s"] = int(self.max_utterance_var.get() or 10)
        self.voice_config["asr"]["partial_interval_ms"] = int(self.partial_interval_var.get() or 150)
        self.voice_config["voice"]["agent_name"] = self.agent_name_var.get() or "Агент"
        self.voice_config["voice"]["voice_name"] = self.voice_name_var.get() or "Microsoft Dmitry"
        self.voice_config["voice"]["startup_voice"] = bool(self.startup_voice_var.get())
        self._save_voice_config(self.voice_config)
        self.settings_status.configure(text="Настройки сохранены.")

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

    def _save_voice_config(self, config: dict[str, dict[str, object]]) -> None:
        CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        existing = {}
        if CONFIG_PATH.exists():
            existing = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8")) or {}
        existing.update(config)
        CONFIG_PATH.write_text(yaml.safe_dump(existing, sort_keys=False, allow_unicode=True), encoding="utf-8")

    def _load_voice_model(self) -> None:
        self.voice_status.configure(text="Модель голоса: загружается...")

        def worker() -> None:
            try:
                VoiceInput(
                    device=get_voice_device(),
                    engine=self.engine_var.get().strip().lower(),
                    model_size=self.model_size_var.get().strip().lower(),
                )
            except Exception as exc:
                self.voice_status.configure(text=f"Модель голоса: ошибка ({exc})")
            else:
                self.voice_status.configure(text="Модель голоса: загружена")

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

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    AgentUI().run()


if __name__ == "__main__":
    main()
