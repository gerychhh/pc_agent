from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk

from core.orchestrator import Orchestrator, sanitize_assistant_text
from core.state import (
    get_voice_device,
    get_voice_engine,
    get_voice_model_size,
    set_voice_device,
    set_voice_engine,
    set_voice_model_size,
)


class AgentUI:
    def __init__(self) -> None:
        self.root = tk.Tk()
        self.root.title("PC Agent")
        self.orchestrator = Orchestrator()
        self.result_queue: queue.Queue[str] = queue.Queue()

        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.chat_frame = ttk.Frame(self.notebook, padding=8)
        self.settings_frame = ttk.Frame(self.notebook, padding=8)
        self.notebook.add(self.chat_frame, text="Chat")
        self.notebook.add(self.settings_frame, text="Settings")

        self._build_chat()
        self._build_settings()

        self.root.after(100, self._poll_results)

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
        engine_label = ttk.Label(self.settings_frame, text="Voice engine")
        engine_label.grid(row=0, column=0, sticky=tk.W, pady=4)
        self.engine_var = tk.StringVar(value=get_voice_engine() or "whisper")
        engine_box = ttk.Combobox(
            self.settings_frame,
            textvariable=self.engine_var,
            values=["vosk", "whisper"],
            state="readonly",
            width=12,
        )
        engine_box.grid(row=0, column=1, sticky=tk.W, pady=4)

        size_label = ttk.Label(self.settings_frame, text="Model size")
        size_label.grid(row=1, column=0, sticky=tk.W, pady=4)
        self.model_size_var = tk.StringVar(value=get_voice_model_size() or "small")
        size_entry = ttk.Entry(self.settings_frame, textvariable=self.model_size_var, width=12)
        size_entry.grid(row=1, column=1, sticky=tk.W, pady=4)

        device_label = ttk.Label(self.settings_frame, text="Voice device index")
        device_label.grid(row=2, column=0, sticky=tk.W, pady=4)
        device_value = get_voice_device()
        self.device_var = tk.StringVar(value="" if device_value is None else str(device_value))
        device_entry = ttk.Entry(self.settings_frame, textvariable=self.device_var, width=12)
        device_entry.grid(row=2, column=1, sticky=tk.W, pady=4)

        save_button = ttk.Button(self.settings_frame, text="Save settings", command=self._save_settings)
        save_button.grid(row=3, column=0, columnspan=2, sticky=tk.W, pady=8)

        self.settings_status = ttk.Label(self.settings_frame, text="")
        self.settings_status.grid(row=4, column=0, columnspan=2, sticky=tk.W)

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
        self.settings_status.configure(text="Settings saved.")

    def run(self) -> None:
        self.root.mainloop()


def main() -> None:
    AgentUI().run()


if __name__ == "__main__":
    main()
