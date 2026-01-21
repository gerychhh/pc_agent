from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass

import sounddevice as sd

from core.app_search_paths import (
    default_search_paths,
    load_search_paths,
    normalize_search_paths,
    save_search_paths,
)
from core.config import (
    SCREENSHOT_DIR,
    VOICE_DEFAULT_ENABLED,
    VOICE_ENGINE,
    VOICE_NAME,
    VOICE_RATE,
    VOICE_VOLUME,
)
from core.orchestrator import Orchestrator, sanitize_assistant_text
from core.debug import set_debug
from core.voice import VoiceInput
from core.state import (
    clear_state,
    get_active_app,
    get_active_file,
    get_active_url,
    get_voice_engine,
    get_voice_model_size,
    get_voice_device,
    load_state,
    set_active_file,
    set_voice_engine,
    set_voice_model_size,
    set_voice_device,
)
from core.interaction_memory import delete_route, find_similar_routes, get_route, record_history, set_route
from core.llm_client import LLMClient
from core.config import FAST_MODEL


HELP_TEXT = """
Commands:
  /help    Show this help message
  /exit    Exit the application
  /reset   Reset conversation context
  /debug   Toggle debug logging
  /screens List recent screenshots
  /active  Show current active file/url/app
  /files   List recent files
  /urls    List recent URLs
  /apps    List recent apps
  /use     Set active file by index or path
  /clear   Clear active state
  /voice models        List voice recognition models
  /voice model <engine> <size>  Set voice model (vosk/whisper + small/full)
""".strip()


class InputManager:
    def __init__(self, voice_enabled: bool) -> None:
        self.text_queue: queue.Queue[str | None] = queue.Queue()
        self.voice_queue: queue.Queue[dict[str, str]] = queue.Queue()
        self._text_thread = threading.Thread(target=self._text_loop, daemon=True)
        self._text_thread.start()
        self._voice_thread: threading.Thread | None = None
        self._voice_stop = threading.Event()
        self.voice_enabled = False
        self.voice_input: VoiceInput | None = None
        self.set_voice_enabled(voice_enabled)

    def _text_loop(self) -> None:
        while True:
            try:
                line = input()
            except (EOFError, KeyboardInterrupt):
                self.text_queue.put(None)
                break
            self.text_queue.put(line)

    def _init_voice_input(self) -> VoiceInput:
        device_idx = get_voice_device()
        engine = get_voice_engine() or VOICE_ENGINE
        model_size = get_voice_model_size()
        return VoiceInput(device=device_idx, engine=engine, model_size=model_size)

    def _voice_loop(self) -> None:
        while not self._voice_stop.is_set():
            try:
                if self.voice_input is None:
                    self.voice_input = self._init_voice_input()
                text = self.voice_input.listen_once()
            except Exception as exc:
                self.voice_queue.put({"type": "voice_error", "error": str(exc)})
                time.sleep(0.2)
                continue
            if self._voice_stop.is_set():
                break
            if text:
                self.voice_queue.put({"type": "voice", "text": text})

    def set_voice_enabled(self, enabled: bool) -> None:
        if enabled and not self.voice_enabled:
            self._voice_stop.clear()
            self.voice_enabled = True
            self._voice_thread = threading.Thread(target=self._voice_loop, daemon=True)
            self._voice_thread.start()
        elif not enabled and self.voice_enabled:
            self._voice_stop.set()
            self.voice_enabled = False
            self.voice_input = None
            self._voice_thread = None

    def reset_voice_input(self) -> None:
        self.voice_input = None

    def get_event(self, timeout: float = 0.1) -> dict[str, str] | None:
        try:
            event = self.voice_queue.get_nowait()
            return event
        except queue.Empty:
            pass
        try:
            line = self.text_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if line is None:
            return {"type": "eof"}
        return {"type": "text", "text": line}


@dataclass
class PendingTask:
    original_query: str
    resolved_query: str
    force_llm: bool
    thread: threading.Thread
    queue: queue.Queue[dict[str, str]]
    cancel: threading.Event


def list_screenshots() -> None:
    screenshots = sorted(SCREENSHOT_DIR.glob("*.png"), key=os.path.getmtime, reverse=True)
    if not screenshots:
        print("No screenshots yet.")
        return
    print("Recent screenshots:")
    for shot in screenshots[:10]:
        print(f" - {shot}")


def _print_recent(label: str, items: list[str]) -> None:
    if not items:
        print(f"No {label}.")
        return
    print(f"Recent {label}:")
    for idx, item in enumerate(items[:10], start=1):
        print(f"{idx}. {item}")


def _escape_powershell(text: str) -> str:
    return text.replace("'", "''")


def _handle_debug_command(raw: str) -> None:
    parts = raw.split()
    if len(parts) == 1:
        set_debug(True)
        print("Debug enabled.")
        return
    if parts[1].lower() in ("off", "0", "false"):
        set_debug(False)
        print("Debug disabled.")
        return
    set_debug(True)
    print("Debug enabled.")


def _should_speak(response: str) -> bool:
    if not response:
        return False
    if "returncode=" in response:
        return False
    if response.startswith(("✅", "❌")):
        return False
    if len(response) > 200:
        return False
    return True


def _parse_yes_no(text: str) -> bool | None:
    normalized = text.strip().lower()
    positive_tokens = {"да", "ага", "верно", "yes", "y", "точно", "правильно", "конечно"}
    negative_tokens = {"нет", "не", "неверно", "no", "n", "неправильно", "не то"}
    words = {part.strip(".,!?") for part in normalized.split()}
    if words & positive_tokens:
        return True
    if words & negative_tokens:
        return False
    return None


def _parse_cancel(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized in {
        "отмена",
        "откажись",
        "забудь",
        "не надо",
        "стоп",
        "стопит",
        "остановись",
        "остановить",
        "стой",
        "хватит",
    }


def _resolve_request(user_text: str) -> tuple[str, bool]:
    similar = find_similar_routes(user_text, limit=3)
    if similar and similar[0]["score"] >= 0.92:
        return similar[0]["resolved"], False
    prompt = (
        "Ты классификатор пользовательских запросов для Windows-агента. "
        "Определи тип запроса и верни JSON:\n"
        "{\n"
        "  \"kind\": \"app|site|youtube|other\",\n"
        "  \"resolved\": \"переформулированный запрос\",\n"
        "  \"force_llm\": true|false\n"
        "}\n"
        "Правила:\n"
        "- app: если пользователь хочет открыть приложение (например, 'открой блокнот').\n"
        "- site: если пользователь хочет открыть сайт/страницу.\n"
        "- youtube: если запрос про поиск/видео/управление YouTube.\n"
        "- other: любая другая задача.\n"
        "- resolved должен быть коротким и ясным действием на русском.\n"
        "- force_llm=true для site/youtube/other, false для app.\n"
        "- Если есть похожие примеры, адаптируй resolved по аналогии.\n"
        "- Верни только JSON."
    )
    examples = "\n".join(
        f"- user: {item['query']} -> resolved: {item['resolved']} (score={item['score']:.2f})" for item in similar
    )
    try:
        response = LLMClient().chat(
            [
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Запрос: {user_text}\nПохожие примеры:\n{examples or 'нет'}"},
            ],
            tools=[],
            model_name=FAST_MODEL,
            tool_choice="none",
        )
    except Exception:
        return user_text, False
    content = response.choices[0].message.content or ""
    try:
        data = json.loads(content.strip())
    except json.JSONDecodeError:
        return user_text, False
    resolved = data.get("resolved")
    force_llm = bool(data.get("force_llm"))
    if isinstance(resolved, str) and resolved.strip():
        return resolved.strip(), force_llm
    return user_text, force_llm


def _format_prompt(text: str) -> str:
    return text if text.endswith(" ") else f"{text} "


def _extract_voice_command(text: str, wake_name: str | None) -> str | None:
    if not wake_name:
        return text
    wake = wake_name.strip().lower()
    if not wake:
        return text
    normalized = text.strip().lower()
    prefixes = (wake, f"эй {wake}", f"hey {wake}")
    for prefix in prefixes:
        if normalized == prefix:
            return ""
        if normalized.startswith(prefix):
            remainder = text[len(prefix) :].lstrip(" ,.!?:;—-")
            return remainder
    return None


def _is_garbage_voice(text: str) -> bool:
    trimmed = text.strip().lower()
    if len(trimmed) < 3:
        return True
    parts = trimmed.split()
    if len(parts) == 1:
        filler = {
            "ээ",
            "эм",
            "мм",
            "угу",
            "ага",
            "ну",
            "да",
            "нет",
            "ок",
            "окей",
            "okay",
            "хм",
        }
        if parts[0] in filler:
            return True
    return False


def speak_text(text: str) -> None:
    safe_text = _escape_powershell(text)
    command = (
        "Add-Type -AssemblyName System.Speech; "
        "$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"try {{ $synth.SelectVoice('{VOICE_NAME}') }} catch {{}}; "
        f"$synth.Rate = {VOICE_RATE}; "
        f"$synth.Volume = {VOICE_VOLUME}; "
        "$synth.Speak("
        f"'{safe_text}')"
    )
    subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )


def ensure_app_search_paths() -> None:
    existing = load_search_paths()
    if existing:
        return
    print("Похоже, список папок для поиска приложений пуст.")
    print("Укажи папки через ';' (например: C:\\\\Program Files;C:\\\\Program Files (x86)).")
    print("Нажми Enter, чтобы использовать рекомендованные пути по умолчанию.")
    raw = input("Paths> ").strip()
    if raw:
        paths = [item.strip() for item in raw.split(";") if item.strip()]
    else:
        paths = default_search_paths()
    normalized = normalize_search_paths(paths)
    if not normalized:
        normalized = default_search_paths()
    save_search_paths(normalized)
    print("Список папок сохранен:")
    for path in normalized:
        print(f" - {path}")


def main() -> None:
    print("PC Agent CLI. Type /help for commands.")
    print("Tip: включить голосовой ввод → /voice on (и проверь микрофон: /voice devices)")
    voice_wake_name = input("Имя для обращения к агенту (например, 'Агент')> ").strip()
    ensure_app_search_paths()
    set_debug(os.getenv("PC_AGENT_DEBUG", "0") == "1")
    orchestrator = Orchestrator()
    voice_enabled = VOICE_DEFAULT_ENABLED
    input_manager = InputManager(voice_enabled)
    prompt_state = "command"
    prompt_shown = False
    prompt_spoken = False
    pending_task: PendingTask | None = None
    pending_confirmation: dict[str, str] | None = None
    queued_command: str | None = None

    def show_prompt(text: str, speak: bool = False) -> None:
        nonlocal prompt_shown, prompt_spoken
        print(_format_prompt(text), end="", flush=True)
        prompt_shown = True
        if speak and voice_enabled and not prompt_spoken:
            try:
                speak_text(text.rstrip())
            except Exception:
                pass
            prompt_spoken = True

    def start_task(original_query: str, resolved_query: str, force_llm: bool) -> None:
        nonlocal pending_task
        result_queue: queue.Queue[dict[str, str]] = queue.Queue()
        cancel_event = threading.Event()

        def runner() -> None:
            try:
                response_text = orchestrator.run(resolved_query, stateless=voice_enabled, force_llm=force_llm)
                result_queue.put({"type": "result", "response": response_text})
            except Exception as exc:
                result_queue.put({"type": "error", "error": str(exc)})

        thread = threading.Thread(target=runner, daemon=True)
        pending_task = PendingTask(
            original_query=original_query,
            resolved_query=resolved_query,
            force_llm=force_llm,
            thread=thread,
            queue=result_queue,
            cancel=cancel_event,
        )
        thread.start()

    while True:
        if pending_task:
            task_queue = pending_task.queue
            try:
                task_result = task_queue.get_nowait()
            except queue.Empty:
                task_result = None
            if task_result:
                if pending_task.cancel.is_set():
                    pending_task = None
                    prompt_state = "command"
                    prompt_shown = False
                    prompt_spoken = False
                else:
                    if task_result["type"] == "error":
                        output = task_result["error"]
                    else:
                        output = sanitize_assistant_text(task_result["response"])
                    if not output:
                        output = "(no output)"
                    print(f"Agent> {output}")
                    if voice_enabled and _should_speak(output):
                        try:
                            speak_text(output)
                        except Exception:
                            pass
                    pending_confirmation = {
                        "original_query": pending_task.original_query,
                        "resolved_query": pending_task.resolved_query,
                        "response": output,
                    }
                    pending_task = None
                    prompt_state = "confirm"
                    prompt_shown = False
                    prompt_spoken = False
                continue

        if queued_command and not pending_task and prompt_state == "command":
            user_input = queued_command
            queued_command = None
        else:
            if prompt_state == "command":
                show_prompt("You>", speak=False)
            elif prompt_state == "confirm":
                show_prompt("Я верно всё сделал? (да/нет)>", speak=True)
            elif prompt_state == "correction":
                show_prompt("Что нужно было сделать? Опиши подробнее (или 'отмена')>", speak=True)

            event = input_manager.get_event(timeout=0.1)
            if event is None:
                continue
            if event["type"] == "eof":
                print("\nExiting.")
                break
            if event["type"] == "voice_error":
                print(f"Voice error: {event.get('error', '')}")
                input_manager.set_voice_enabled(False)
                voice_enabled = False
                prompt_shown = False
                prompt_spoken = False
                continue
            if event["type"] == "voice":
                voice_text = event.get("text", "")
                if _is_garbage_voice(voice_text):
                    print("Не расслышал, повтори.")
                    prompt_shown = False
                    prompt_spoken = False
                    continue
                normalized = voice_text.strip().lower()
                if normalized in {"выключи голос", "voice off"}:
                    voice_enabled = False
                    input_manager.set_voice_enabled(False)
                    print("Voice mode disabled.")
                    prompt_shown = False
                    prompt_spoken = False
                    continue
                if prompt_state in {"confirm", "correction"} or _parse_cancel(voice_text):
                    user_input = voice_text.strip()
                    print(f"You(voice)> {user_input}")
                else:
                    command = _extract_voice_command(voice_text, voice_wake_name)
                    if command is None:
                        prompt_shown = False
                        prompt_spoken = False
                        continue
                    if not command:
                        if pending_task:
                            pending_task.cancel.set()
                            pending_task = None
                        print("Да, слушаю.")
                        if voice_enabled:
                            try:
                                speak_text("Да, слушаю.")
                            except Exception:
                                pass
                        prompt_shown = False
                        prompt_spoken = False
                        continue
                    print(f"You(voice)> {command}")
                    user_input = command
            else:
                user_input = event.get("text", "").strip()

        if not user_input:
            prompt_shown = False
            prompt_spoken = False
            continue

        if _parse_cancel(user_input):
            if pending_task:
                pending_task.cancel.set()
            pending_task = None
            pending_confirmation = None
            prompt_state = "command"
            queued_command = None
            print("Остановлено.")
            if voice_enabled:
                try:
                    speak_text("Остановлено.")
                except Exception:
                    pass
            prompt_shown = False
            prompt_spoken = False
            continue

        if pending_task:
            queued_command = user_input
            pending_task.cancel.set()
            print("Остановлено.")
            if voice_enabled:
                try:
                    speak_text("Остановлено.")
                except Exception:
                    pass
            prompt_shown = False
            prompt_spoken = False
            continue

        if prompt_state == "confirm":
            verdict = _parse_yes_no(user_input)
            if verdict is None:
                print("Ответь 'да' или 'нет'.")
                prompt_shown = False
                prompt_spoken = False
                continue
            if verdict:
                if pending_confirmation:
                    original_query = pending_confirmation["original_query"]
                    resolved_query = pending_confirmation["resolved_query"]
                    response = pending_confirmation["response"]
                    if resolved_query != original_query:
                        set_route(original_query, resolved_query)
                    record_history(original_query, response, resolved_query)
                pending_confirmation = None
                prompt_state = "command"
                prompt_shown = False
                prompt_spoken = False
                continue
            prompt_state = "correction"
            prompt_shown = False
            prompt_spoken = False
            continue

        if prompt_state == "correction":
            correction = user_input.strip()
            if _parse_cancel(correction):
                if pending_confirmation:
                    delete_route(pending_confirmation["original_query"])
                print("Команда забыта.")
                pending_confirmation = None
                prompt_state = "command"
                prompt_shown = False
                prompt_spoken = False
                continue
            if not correction:
                print("Нужно описание корректного действия.")
                prompt_shown = False
                prompt_spoken = False
                continue
            if not pending_confirmation:
                prompt_state = "command"
                prompt_shown = False
                prompt_spoken = False
                continue
            if voice_enabled:
                try:
                    speak_text("Думаю над новой задачей.")
                except Exception:
                    pass
            resolved_correction, force_llm = _resolve_request(correction)
            set_route(pending_confirmation["original_query"], resolved_correction)
            start_task(pending_confirmation["original_query"], resolved_correction, force_llm)
            prompt_state = "command"
            prompt_shown = False
            prompt_spoken = False
            continue

        if user_input == "/help":
            print(HELP_TEXT)
            prompt_shown = False
            prompt_spoken = False
            continue
        if user_input == "/exit":
            print("Goodbye.")
            break
        if user_input == "/reset":
            orchestrator.reset()
            print("Context reset.")
            prompt_shown = False
            prompt_spoken = False
            continue
        if user_input.startswith("/debug"):
            _handle_debug_command(user_input)
            prompt_shown = False
            prompt_spoken = False
            continue
        if user_input == "/active":
            active_file = get_active_file()
            active_url = get_active_url()
            active_app = get_active_app()
            print(f"Active file: {active_file or '(none)'}")
            print(f"Active url: {active_url or '(none)'}")
            print(f"Active app: {active_app or '(none)'}")
            prompt_shown = False
            prompt_spoken = False
            continue
        if user_input == "/files":
            state = load_state()
            _print_recent("files", state.get("recent_files", []))
            prompt_shown = False
            prompt_spoken = False
            continue
        if user_input == "/urls":
            state = load_state()
            _print_recent("urls", state.get("recent_urls", []))
            prompt_shown = False
            prompt_spoken = False
            continue
        if user_input == "/apps":
            state = load_state()
            _print_recent("apps", state.get("recent_apps", []))
            prompt_shown = False
            prompt_spoken = False
            continue
        if user_input.startswith("/use"):
            raw = user_input[len("/use") :].strip()
            if not raw:
                print("Usage: /use <number|path>")
                prompt_shown = False
                prompt_spoken = False
                continue
            if raw.isdigit():
                state = load_state()
                index = int(raw)
                recent_files = state.get("recent_files", [])
                if index < 1 or index > len(recent_files):
                    print("Invalid file number.")
                    prompt_shown = False
                    prompt_spoken = False
                    continue
                selected = recent_files[index - 1]
                set_active_file(selected)
                print(f"Active file set: {selected}")
                prompt_shown = False
                prompt_spoken = False
                continue
            set_active_file(raw)
            print(f"Active file set: {raw}")
            prompt_shown = False
            prompt_spoken = False
            continue
        if user_input == "/clear":
            clear_state()
            print("State cleared.")
            prompt_shown = False
            prompt_spoken = False
            continue
        if user_input == "/screens":
            list_screenshots()
            prompt_shown = False
            prompt_spoken = False
            continue
        if user_input == "/voice devices":
            try:
                devices = sd.query_devices()
                print("Audio devices:")
                for i, d in enumerate(devices):
                    name = d.get("name")
                    ins = d.get("max_input_channels")
                    outs = d.get("max_output_channels")
                    print(f"  {i}: {name} (in={ins}, out={outs})")
            except Exception as exc:
                print(f"Cannot query devices: {exc}")
            prompt_shown = False
            prompt_spoken = False
            continue
        if user_input.startswith("/voice device"):
            parts = user_input.split()
            if len(parts) != 3 or not parts[2].isdigit():
                print("Usage: /voice device <index>")
                prompt_shown = False
                prompt_spoken = False
                continue
            set_voice_device(int(parts[2]))
            input_manager.reset_voice_input()
            print(f"Voice input device set to {parts[2]} (re-init).")
            prompt_shown = False
            prompt_spoken = False
            continue
        if user_input == "/voice models":
            print("Voice recognition models:")
            print("  vosk: small | full (uses local models folder)")
            print("  whisper: small | full (maps to base)")
            print("Use: /voice model <engine> <size>")
            prompt_shown = False
            prompt_spoken = False
            continue
        if user_input.startswith("/voice model"):
            parts = user_input.split()
            if len(parts) != 4:
                print("Usage: /voice model <engine> <size>")
                prompt_shown = False
                prompt_spoken = False
                continue
            engine = parts[2].lower()
            size = parts[3].lower()
            if engine not in {"vosk", "whisper"}:
                print("Engine must be 'vosk' or 'whisper'.")
                prompt_shown = False
                prompt_spoken = False
                continue
            if size not in {"small", "full", "base", "medium", "large"}:
                print("Size must be small/full (or base/medium/large for whisper).")
                prompt_shown = False
                prompt_spoken = False
                continue
            normalized_size = "full" if size in {"full"} else size
            set_voice_engine(engine)
            set_voice_model_size(normalized_size)
            input_manager.reset_voice_input()
            print(f"Voice model set: engine={engine}, size={normalized_size} (re-init).")
            prompt_shown = False
            prompt_spoken = False
            continue

        if user_input.startswith("/voice"):
            if user_input == "/voice" or user_input.endswith("on"):
                voice_enabled = True
                input_manager.set_voice_enabled(True)
                print("Voice mode enabled.")
            elif user_input.endswith("off"):
                voice_enabled = False
                input_manager.set_voice_enabled(False)
                print("Voice mode disabled.")
            else:
                print("Usage: /voice [on|off]")
            prompt_shown = False
            prompt_spoken = False
            continue

        original_query = user_input
        resolved_query = get_route(user_input)
        force_llm = False
        if not resolved_query:
            if voice_enabled:
                try:
                    speak_text("Думаю над новой задачей.")
                except Exception:
                    pass
            resolved_query, force_llm = _resolve_request(user_input)
        start_task(original_query, resolved_query, force_llm)
        prompt_shown = False
        prompt_spoken = False


if __name__ == "__main__":
    main()
