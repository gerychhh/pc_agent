from __future__ import annotations

import os

from core.app_search_paths import (
    default_search_paths,
    load_search_paths,
    normalize_search_paths,
    save_search_paths,
)
from core.config import SCREENSHOT_DIR, VOICE_DEFAULT_ENABLED
from core.orchestrator import Orchestrator, sanitize_assistant_text
from core.voice import VoiceInput


HELP_TEXT = """
Commands:
  /help    Show this help message
  /exit    Exit the application
  /reset   Reset conversation context
  /screens List recent screenshots
""".strip()


def list_screenshots() -> None:
    screenshots = sorted(SCREENSHOT_DIR.glob("*.png"), key=os.path.getmtime, reverse=True)
    if not screenshots:
        print("No screenshots yet.")
        return
    print("Recent screenshots:")
    for shot in screenshots[:10]:
        print(f" - {shot}")


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
    ensure_app_search_paths()
    orchestrator = Orchestrator()
    voice_enabled = VOICE_DEFAULT_ENABLED
    voice_input: VoiceInput | None = None

    while True:
        if voice_enabled:
            print('🎤 Voice mode: speak your command (say "стоп" to exit voice mode)...')
            try:
                if voice_input is None:
                    voice_input = VoiceInput()
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
            normalized = voice_text.strip().lower()
            if normalized in {"стоп", "выключи голос", "stop"}:
                voice_enabled = False
                print("Voice mode disabled.")
                continue
            print(f"You(voice)> {voice_text}")
            user_input = voice_text
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
        if user_input == "/screens":
            list_screenshots()
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

        response = orchestrator.run(user_input)
        output = sanitize_assistant_text(response)
        if not output:
            output = "(no output)"
        print(f"Agent> {output}")


if __name__ == "__main__":
    main()
