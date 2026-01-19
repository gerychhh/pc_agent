from __future__ import annotations

import shutil
import urllib.request
import zipfile
from pathlib import Path

MODEL_URL = "https://alphacephei.com/vosk/models/vosk-model-small-ru-0.22.zip"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"
ZIP_PATH = MODELS_DIR / "vosk-model-small-ru-0.22.zip"


def download_file(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url) as response, dest.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def extract_zip(zip_path: Path, dest_dir: Path) -> Path:
    with zipfile.ZipFile(zip_path, "r") as archive:
        archive.extractall(dest_dir)
    extracted_dir = dest_dir / "vosk-model-small-ru-0.22"
    return extracted_dir


def main() -> None:
    print(f"Downloading {MODEL_URL}")
    download_file(MODEL_URL, ZIP_PATH)
    model_path = extract_zip(ZIP_PATH, MODELS_DIR)
    print(f"Model extracted to: {model_path}")


if __name__ == "__main__":
    main()
