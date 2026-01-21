from __future__ import annotations

import json
import os
import subprocess

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
    get_voice_device,
    load_state,
    set_active_file,
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
""".strip()


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
    if normalized in {"да", "ага", "верно", "yes", "y"}:
        return True
    if normalized in {"нет", "не", "неверно", "no", "n", "неправильно", "не то"}:
        return False
    return None


def _parse_cancel(text: str) -> bool:
    normalized = text.strip().lower()
    return normalized in {"отмена", "откажись", "забудь", "не надо", "стоп"}


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


def _prompt_text(prompt: str, voice_enabled: bool, voice_input: VoiceInput | None) -> str | None:
    if voice_enabled and voice_input:
        print(prompt, end="")
        try:
            speak_text(prompt.rstrip())
        except Exception:
            pass
        return voice_input.listen_once()
    return input(prompt).strip()


def _confirm_request(
    original_query: str,
    resolved_query: str,
    response: str,
    orchestrator: Orchestrator,
    voice_enabled: bool,
    voice_input: VoiceInput | None,
) -> None:
    while True:
        answer = _prompt_text("Я верно всё сделал? (да/нет)> ", voice_enabled, voice_input)
        if answer is None:
            print("Не расслышал, повтори.")
            continue
        answer = answer.strip()
        verdict = _parse_yes_no(answer)
        if verdict is None:
            print("Ответь 'да' или 'нет'.")
            continue
        if verdict:
            if resolved_query != original_query:
                set_route(original_query, resolved_query)
            record_history(original_query, response, resolved_query)
            return
        correction = _prompt_text(
            "Что нужно было сделать? Опиши подробнее (или 'отмена')> ",
            voice_enabled,
            voice_input,
        )
        if correction is None:
            print("Не расслышал, повтори.")
            continue
        correction = correction.strip()
        if _parse_cancel(correction):
            delete_route(original_query)
            print("Команда забыта.")
            return
        if not correction:
            print("Нужно описание корректного действия.")
            continue
        resolved_correction, force_llm = _resolve_request(correction)
        set_route(original_query, resolved_correction)
        response_text = orchestrator.run(resolved_correction, stateless=voice_enabled, force_llm=force_llm)
        output = sanitize_assistant_text(response_text)
        if not output:
            output = "(no output)"
        print(f"Agent> {output}")
        response = output
        resolved_query = resolved_correction


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
    voice_input: VoiceInput | None = None

    while True:
        if voice_enabled:
            try:
                if voice_input is None:
                    device_idx = get_voice_device()
                    voice_input = VoiceInput(device=device_idx)
                voice_text = voice_input.listen_once()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break
            except Exception as exc:
                print(f"Voice error: {exc}")
                voice_enabled = False
                continue
            if not voice_text:
                continue
            if _is_garbage_voice(voice_text):
                print("Не расслышал, повтори.")
                continue
            normalized = voice_text.strip().lower()
            if normalized in {"стоп", "выключи голос", "stop"}:
                voice_enabled = False
                print("Voice mode disabled.")
                continue
            command = _extract_voice_command(voice_text, voice_wake_name)
            if command is None:
                continue
            if not command:
                print("Скажи команду после имени.")
                continue
            print(f"You(voice)> {command}")
            user_input = command
        else:
            try:
                user_input = input("You> ").strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

        if not user_input:
            continue

        if user_input == "/help":
            print(HELP_TEXT)
            continue
        if user_input == "/exit":
            print("Goodbye.")
            break
        if user_input == "/reset":
            orchestrator.reset()
            print("Context reset.")
            continue
        if user_input.startswith("/debug"):
            _handle_debug_command(user_input)
            continue
        if user_input == "/active":
            active_file = get_active_file()
            active_url = get_active_url()
            active_app = get_active_app()
            print(f"Active file: {active_file or '(none)'}")
            print(f"Active url: {active_url or '(none)'}")
            print(f"Active app: {active_app or '(none)'}")
            continue
        if user_input == "/files":
            state = load_state()
            _print_recent("files", state.get("recent_files", []))
            continue
        if user_input == "/urls":
            state = load_state()
            _print_recent("urls", state.get("recent_urls", []))
            continue
        if user_input == "/apps":
            state = load_state()
            _print_recent("apps", state.get("recent_apps", []))
            continue
        if user_input.startswith("/use"):
            raw = user_input[len("/use") :].strip()
            if not raw:
                print("Usage: /use <number|path>")
                continue
            if raw.isdigit():
                state = load_state()
                index = int(raw)
                recent_files = state.get("recent_files", [])
                if index < 1 or index > len(recent_files):
                    print("Invalid file number.")
                    continue
                selected = recent_files[index - 1]
                set_active_file(selected)
                print(f"Active file set: {selected}")
                continue
            set_active_file(raw)
            print(f"Active file set: {raw}")
            continue
        if user_input == "/clear":
            clear_state()
            print("State cleared.")
            continue
        if user_input == "/screens":
            list_screenshots()
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
            continue
        if user_input.startswith("/voice device"):
            parts = user_input.split()
            if len(parts) != 3 or not parts[2].isdigit():
                print("Usage: /voice device <index>")
                continue
            set_voice_device(int(parts[2]))
            voice_input = None
            print(f"Voice input device set to {parts[2]} (re-init).")
            continue

        if user_input.startswith("/voice"):
            if user_input == "/voice" or user_input.endswith("on"):
                voice_enabled = True
                print("Voice mode enabled.")
            elif user_input.endswith("off"):
                voice_enabled = False
                print("Voice mode disabled.")
            else:
                print("Usage: /voice [on|off]")
            continue

        original_query = user_input
        resolved_query = get_route(user_input)
        force_llm = False
        if not resolved_query:
            resolved_query, force_llm = _resolve_request(user_input)
        response = orchestrator.run(resolved_query, stateless=voice_enabled, force_llm=force_llm)
        output = sanitize_assistant_text(response)
        if not output:
            output = "(no output)"
        print(f"Agent> {output}")
        if voice_enabled and _should_speak(output):
            try:
                speak_text(output)
            except Exception:
                pass
        _confirm_request(original_query, resolved_query, output, orchestrator, voice_enabled, voice_input)


if __name__ == "__main__":
    main()
