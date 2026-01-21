from __future__ import annotations

import subprocess
import sys
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


def _openwakeword_requires_config() -> bool:
    try:
        result = subprocess.run(
            [sys.executable, "-m", "openwakeword.train", "-h"],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    output = f"{result.stdout}\n{result.stderr}"
    return "--training_config" in output


def _prompt_training_config() -> str:
    while True:
        raw = input("Путь к training_config.yaml: ").strip()
        if raw:
            return raw
        print("Нужен путь к training_config.yaml.")


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
    requires_config = _openwakeword_requires_config()

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
            if requires_config:
                config_path = Path("configs/training_config.yaml")
                if not config_path.exists():
                    config_path = Path(_prompt_training_config())
                train_cmd = f"python -m openwakeword.train --training_config {config_path} --train_model"
                exit_code = _run_train_cmd(train_cmd)
            else:
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
