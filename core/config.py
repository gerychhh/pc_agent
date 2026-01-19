from __future__ import annotations

import os
from pathlib import Path

BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
API_KEY = os.getenv("LMSTUDIO_API_KEY", "not-needed")
MODEL_NAME = os.getenv("MODEL_NAME", "local-model")

MODE = "script"
DEBUG = os.getenv("DEBUG") == "1"
MAX_RETRIES = 2
TIMEOUT_SEC = 30

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"
VOICE_DEFAULT_ENABLED = os.getenv("VOICE", "0") == "1"
VOSK_MODEL_DIR = Path(os.getenv("VOSK_MODEL_DIR", PROJECT_ROOT / "models" / "vosk-model-small-ru-0.22"))
VOICE_SAMPLE_RATE = int(os.getenv("VOICE_SAMPLE_RATE", "16000"))
VOICE_NAME = os.getenv("VOICE_NAME", "Microsoft Pavel")
VOICE_RATE = int(os.getenv("VOICE_RATE", "2"))
VOICE_VOLUME = int(os.getenv("VOICE_VOLUME", "100"))

LOG_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
