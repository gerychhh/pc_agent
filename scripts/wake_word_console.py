from __future__ import annotations

import subprocess
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import importlib.util
import numpy as np
import sounddevice as sd


@dataclass(frozen=True)
class AudioSettings:
    sample_rate: int = 16000
    channels: int = 1


def _write_wav(path: Path, audio: np.ndarray, settings: AudioSettings) -> None:
    audio_int16 = np.clip(audio * 32768.0, -32768, 32767).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(settings.channels)
        wf.setsampwidth(2)
        wf.setframerate(settings.sample_rate)
        wf.writeframes(audio_int16.tobytes())


def _record_sample(duration_s: float, settings: AudioSettings) -> np.ndarray:
    frames = int(duration_s * settings.sample_rate)
    audio = sd.rec(frames, samplerate=settings.sample_rate, channels=settings.channels, dtype="float32")
    sd.wait()
    return audio[:, 0] if audio.ndim > 1 else audio


def _countdown(seconds: int) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"... {remaining}")
        time.sleep(1)


def _prompt_int(prompt: str, default: int) -> int:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    return int(raw)


def _prompt_float(prompt: str, default: float) -> float:
    raw = input(f"{prompt} [{default}]: ").strip()
    if not raw:
        return default
    return float(raw)


def _openwakeword_available() -> bool:
    return importlib.util.find_spec("openwakeword") is not None


def collect_samples(output_dir: Path, label: str, settings: AudioSettings) -> None:
    count = _prompt_int("Сколько записей сделать", 20)
    duration_s = _prompt_float("Длительность записи (сек)", 1.2)
    prompt = "Произнесите ключевую фразу." if label == "positive" else "Говорите обычные слова или тишина."

    output_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(output_dir.glob("*.wav")))
    for idx in range(count):
        print(f"[{label}] Sample {idx + 1}/{count}: {prompt}")
        _countdown(2)
        audio = _record_sample(duration_s, settings)
        filename = f"{label}_{existing + idx + 1:04d}.wav"
        path = output_dir / filename
        _write_wav(path, audio, settings)
        time.sleep(0.2)
    print(f"Saved {count} samples to {output_dir}")


def train_model(training_config: Path) -> None:
    cmd = [
        "python",
        "-m",
        "openwakeword.train",
        "--training_config",
        str(training_config),
        "--generate_clips",
        "--overwrite",
        "--augment_clips",
        "--train_model",
    ]
    print("Running:")
    print(" ".join(cmd))
    confirm = input("Запустить обучение? (y/N): ").strip().lower()
    if confirm != "y":
        print("Отменено.")
        return
    subprocess.run(cmd, check=False)


def test_model(model_path: Path, samples_dir: Path) -> None:
    if not _openwakeword_available():
        print("openwakeword не установлен. Установите зависимости для теста модели.")
        return
    from openwakeword.model import Model

    if not model_path.exists():
        print(f"Модель не найдена: {model_path}")
        return
    wavs = sorted(samples_dir.glob("*.wav"))
    if not wavs:
        print(f"Нет .wav файлов в {samples_dir}")
        return
    model = Model(wakeword_models=[str(model_path)])
    print(f"Testing on {len(wavs)} samples...")
    for wav_path in wavs:
        with wave.open(str(wav_path), "rb") as wf:
            frames = wf.readframes(wf.getnframes())
            audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        scores = model.predict(audio)
        top_score = max(scores.values()) if scores else 0.0
        print(f"{wav_path.name}: {top_score:.3f}")


def install_model(source_path: Path, config_dir: Path) -> None:
    target_dir = config_dir / "models"
    target_dir.mkdir(parents=True, exist_ok=True)
    target_path = target_dir / "agent.onnx"
    if not source_path.exists():
        print(f"Модель не найдена: {source_path}")
        return
    target_path.write_bytes(source_path.read_bytes())
    print(f"Модель скопирована в {target_path}")


def run_doctor(script_path: Path) -> None:
    if not script_path.exists():
        print(f"Doctor-скрипт не найден: {script_path}")
        return
    cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script_path)]
    subprocess.run(cmd, check=False)


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    data_root = repo_root / "data"
    training_config = repo_root / "configs" / "training_config.yaml"
    model_path = repo_root / "models" / "agent.onnx"
    doctor_script = repo_root / "scripts" / "doctor_openwakeword_train.ps1"
    voice_agent_dir = repo_root / "voice_agent"

    settings = AudioSettings()

    while True:
        print("\nWake-word console menu")
        print("1) Записать позитивные примеры")
        print("2) Записать негативные примеры")
        print("3) Обучить модель openWakeWord")
        print("4) Протестировать модель")
        print("5) Установить модель в агента")
        print("6) Запустить doctor")
        print("0) Выход")
        choice = input("> ").strip()

        if choice == "1":
            collect_samples(data_root / "positive", "positive", settings)
        elif choice == "2":
            collect_samples(data_root / "negative", "negative", settings)
        elif choice == "3":
            train_model(training_config)
        elif choice == "4":
            test_model(model_path, data_root / "positive")
        elif choice == "5":
            install_model(model_path, voice_agent_dir)
        elif choice == "6":
            run_doctor(doctor_script)
        elif choice == "0":
            print("Пока!")
            return
        else:
            print("Неизвестная команда.")


if __name__ == "__main__":
    main()
