from __future__ import annotations

import sys
import urllib.request
import zipfile
from pathlib import Path

MODEL_NAME = "vosk-model-ru-0.22"
MODEL_URL = f"https://alphacephei.com/vosk/models/{MODEL_NAME}.zip"


def _print_progress(block_count: int, block_size: int, total_size: int) -> None:
    if total_size <= 0:
        return
    downloaded = min(block_count * block_size, total_size)
    percent = downloaded / total_size * 100
    sys.stdout.write(f"\rDownloading {MODEL_NAME}: {percent:5.1f}%")
    sys.stdout.flush()


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    models_dir = project_root / "models"
    model_dir = models_dir / MODEL_NAME
    zip_path = models_dir / f"{MODEL_NAME}.zip"

    if model_dir.exists():
        print(f"Vosk model already exists at {model_dir}")
        return 0

    models_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {MODEL_NAME} from {MODEL_URL}")
    urllib.request.urlretrieve(MODEL_URL, zip_path, reporthook=_print_progress)
    print()

    print("Extracting model...")
    with zipfile.ZipFile(zip_path, "r") as zip_file:
        zip_file.extractall(models_dir)

    zip_path.unlink(missing_ok=True)
    print(f"Done. Model available at {model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
