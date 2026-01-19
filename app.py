from __future__ import annotations

import os
from pathlib import Path

from core.config import SCREENSHOT_DIR
from core.orchestrator import Orchestrator


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


def main() -> None:
    print("PC Agent CLI. Type /help for commands.")
    orchestrator = Orchestrator()

    while True:
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

        response = orchestrator.run(user_input)
        print(f"Agent> {response}")


if __name__ == "__main__":
    main()
