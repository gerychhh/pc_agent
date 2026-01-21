from __future__ import annotations

import importlib.util
import shutil
import urllib.request
from pathlib import Path

FEATURE_MODELS = {
    "melspectrogram.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/melspectrogram.onnx",
    "embedding_model.onnx": "https://github.com/dscripka/openWakeWord/releases/download/v0.5.1/embedding_model.onnx",
}


def _openwakeword_models_dir() -> Path:
    spec = importlib.util.find_spec("openwakeword")
    if spec is None or spec.origin is None:
        raise FileNotFoundError("openwakeword не установлен в текущем окружении.")
    return Path(spec.origin).parent / "resources" / "models"


def ensure_feature_models() -> None:
    target_dir = _openwakeword_models_dir()
    target_dir.mkdir(parents=True, exist_ok=True)
    for filename, url in FEATURE_MODELS.items():
        target_file = target_dir / filename
        if not target_file.exists():
            urllib.request.urlretrieve(url, target_file)


if __name__ == "__main__":
    ensure_feature_models()
    print("OpenWakeWord feature models are ready.")
