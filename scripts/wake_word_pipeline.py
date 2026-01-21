from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import importlib.util
import numpy as np
import librosa
import scipy.signal
import soundfile as sf
import sounddevice as sd


@dataclass(frozen=True)
class AudioSettings:
    sample_rate: int = 16000
    channels: int = 1


def _write_wav(path: Path, audio: np.ndarray, settings: AudioSettings) -> None:
    audio = np.asarray(audio, dtype=np.float32)
    sf.write(
        file=str(path),
        data=audio,
        samplerate=settings.sample_rate,
        subtype="PCM_16",
    )


def _load_audio(path: Path, target_sr: int) -> np.ndarray:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    audio = scipy.signal.detrend(audio, type="constant")
    return audio.astype(np.float32)


def _record_sample(duration_s: float, settings: AudioSettings) -> np.ndarray:
    frames = int(duration_s * settings.sample_rate)
    audio = sd.rec(frames, samplerate=settings.sample_rate, channels=settings.channels, dtype="float32")
    sd.wait()
    return audio[:, 0] if audio.ndim > 1 else audio


def _countdown(seconds: int) -> None:
    for remaining in range(seconds, 0, -1):
        print(f"... {remaining}")
        time.sleep(1)


def collect_samples(
    output_dir: Path,
    label: str,
    count: int,
    duration_s: float,
    settings: AudioSettings,
    prompt: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata = []
    for idx in range(count):
        print(f"[{label}] Sample {idx + 1}/{count}: {prompt}")
        _countdown(2)
        audio = _record_sample(duration_s, settings)
        filename = f"{label}_{idx + 1:04d}.wav"
        path = output_dir / filename
        _write_wav(path, audio, settings)
        metadata.append({"path": str(path), "label": label, "duration_s": duration_s})
        time.sleep(0.2)
    manifest = output_dir / "manifest.json"
    manifest.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {count} samples to {output_dir}")


def _openwakeword_available() -> bool:
    return importlib.util.find_spec("openwakeword") is not None


def _openwakeword_missing_melspec() -> Path | None:
    if not _openwakeword_available():
        return None
    import openwakeword

    root = Path(openwakeword.__file__).resolve().parent
    melspec = root / "resources" / "models" / "melspectrogram.onnx"
    return melspec if not melspec.exists() else None


def _raise_missing_melspec() -> None:
    missing_path = _openwakeword_missing_melspec()
    if missing_path is None:
        return
    raise SystemExit(
        "В пакете openwakeword отсутствует файл melspectrogram.onnx. "
        "Переустановите пакет или скопируйте модель в "
        f"{missing_path}. Подробнее: voice_agent/WAKE_WORD_TRAINING.md"
    )


def train_model(
    dataset_dir: Path,
    output_model: Path,
    train_cmd: str | None,
) -> None:
    if train_cmd:
        _raise_missing_melspec()
        print(f"Running training command: {train_cmd}")
        result = subprocess.run(train_cmd, shell=True, check=False)
        if result.returncode != 0:
            raise SystemExit(result.returncode)
        return
    if not _openwakeword_available():
        raise SystemExit(
            "openwakeword не установлен. Установите зависимости и запустите с --train-cmd, "
            "либо установите openwakeword и запустите обучение вручную."
        )
    print(
        "Автоматическое обучение требует скрипта openWakeWord. "
        "Передайте --train-cmd с командой обучения (см. docs openWakeWord)."
    )
    raise SystemExit(2)


def test_model(model_path: Path, samples_dir: Path, settings: AudioSettings) -> None:
    if not _openwakeword_available():
        raise SystemExit("openwakeword не установлен. Установите зависимости для теста модели.")
    _raise_missing_melspec()
    from openwakeword.model import Model

    model = Model(wakeword_models=[str(model_path)])
    wavs = sorted(samples_dir.glob("*.wav"))
    if not wavs:
        raise SystemExit(f"Нет .wav файлов в {samples_dir}")
    print(f"Testing on {len(wavs)} samples...")
    for wav_path in wavs:
        audio = _load_audio(wav_path, target_sr=settings.sample_rate)
        scores = model.predict(audio)
        top_score = max(scores.values()) if scores else 0.0
        print(f"{wav_path.name}: {top_score:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Wake-word data collection and training helper.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    collect_parser = subparsers.add_parser("collect", help="Collect audio samples.")
    collect_parser.add_argument("--label", required=True, choices=["positive", "negative"])
    collect_parser.add_argument("--count", type=int, default=50)
    collect_parser.add_argument("--duration-s", type=float, default=1.2)
    collect_parser.add_argument("--output", type=Path, default=Path("data"))
    collect_parser.add_argument("--prompt", default="Произнесите ключевую фразу.")

    train_parser = subparsers.add_parser("train", help="Train a wake-word model.")
    train_parser.add_argument("--dataset", type=Path, default=Path("data"))
    train_parser.add_argument("--output-model", type=Path, default=Path("models/agent.onnx"))
    train_parser.add_argument("--train-cmd", type=str, default=None)

    test_parser = subparsers.add_parser("test", help="Test the trained wake-word model.")
    test_parser.add_argument("--model", type=Path, default=Path("models/agent.onnx"))
    test_parser.add_argument("--samples", type=Path, default=Path("data/positive"))

    full_parser = subparsers.add_parser("full", help="Collect data, train, and test.")
    full_parser.add_argument("--positive-count", type=int, default=150)
    full_parser.add_argument("--negative-count", type=int, default=300)
    full_parser.add_argument("--duration-s", type=float, default=1.2)
    full_parser.add_argument("--data-root", type=Path, default=Path("data"))
    full_parser.add_argument("--train-cmd", type=str, default=None)
    full_parser.add_argument("--output-model", type=Path, default=Path("models/agent.onnx"))

    args = parser.parse_args()
    settings = AudioSettings()

    if args.command == "collect":
        output_dir = args.output / args.label
        collect_samples(output_dir, args.label, args.count, args.duration_s, settings, args.prompt)
        return

    if args.command == "train":
        train_model(args.dataset, args.output_model, args.train_cmd)
        return

    if args.command == "test":
        test_model(args.model, args.samples, settings)
        return

    if args.command == "full":
        data_root: Path = args.data_root
        collect_samples(
            data_root / "positive",
            "positive",
            args.positive_count,
            args.duration_s,
            settings,
            "Произнесите ключевое слово.",
        )
        collect_samples(
            data_root / "negative",
            "negative",
            args.negative_count,
            args.duration_s,
            settings,
            "Говорите обычные слова или тишина.",
        )
        train_model(data_root, args.output_model, args.train_cmd)
        if args.output_model.exists():
            test_model(args.output_model, data_root / "positive", settings)
        return

    raise SystemExit(1)


if __name__ == "__main__":
    main()
