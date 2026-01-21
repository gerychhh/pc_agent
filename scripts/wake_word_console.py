from __future__ import annotations

import subprocess
from pathlib import Path

from wake_word_pipeline import AudioSettings, collect_samples, test_model, train_model


def _prompt_int(label: str, default: int) -> int:
    raw = input(f"{label} [{default}]: ").strip()
    return int(raw) if raw else default


def _prompt_float(label: str, default: float) -> float:
    raw = input(f"{label} [{default}]: ").strip()
    return float(raw) if raw else default


def _prompt_text(label: str, default: str) -> str:
    raw = input(f"{label} [{default}]: ").strip()
    return raw if raw else default


def _prompt_train_cmd(default: str) -> str | None:
    raw = input(f"Команда обучения (Enter = дефолт, 'none' = встроенная) [{default}]: ").strip()
    if not raw:
        return default
    if raw.lower() == "none":
        return None
    return raw


def _run_train_cmd(command: str) -> int:
    print(f"Running training command: {command}")
    result = subprocess.run(command, shell=True, check=False)
    return result.returncode


def _menu() -> str:
    print("\nWake-word console")
    print("1) Записать позитивные примеры")
    print("2) Записать негативные примеры")
    print("3) Обучить модель")
    print("4) Тестировать модель")
    print("5) Выход")
    return input("Выберите действие: ").strip()


def main() -> None:
    settings = AudioSettings()
    data_root = Path("data")
    model_path = Path("models/agent.onnx")
    default_train_cmd = "python -m openwakeword.train --dataset data --output models/agent.onnx"

    while True:
        choice = _menu()

        if choice == "1":
            count = _prompt_int("Количество позитивных примеров", 150)
            duration = _prompt_float("Длительность, сек", 1.2)
            prompt = _prompt_text("Подсказка", "Произнесите ключевую фразу.")
            collect_samples(data_root / "positive", "positive", count, duration, settings, prompt)
            continue

        if choice == "2":
            count = _prompt_int("Количество негативных примеров", 300)
            duration = _prompt_float("Длительность, сек", 1.2)
            prompt = _prompt_text("Подсказка", "Говорите обычные слова или тишина.")
            collect_samples(data_root / "negative", "negative", count, duration, settings, prompt)
            continue

        if choice == "3":
            train_cmd = _prompt_train_cmd(default_train_cmd)
            if train_cmd is None:
                train_model(data_root, model_path, train_cmd)
                continue
            exit_code = _run_train_cmd(train_cmd)
            if exit_code != 0:
                raise SystemExit(exit_code)
            continue

        if choice == "4":
            test_model(model_path, data_root / "positive", settings)
            continue

        if choice == "5":
            print("Выход.")
            return

        if choice:
            print(f"Неизвестная команда: {choice}")


if __name__ == "__main__":
    main()
