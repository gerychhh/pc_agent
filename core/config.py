from __future__ import annotations

import os
from pathlib import Path

BASE_URL = os.getenv("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
API_KEY = os.getenv("LMSTUDIO_API_KEY", "not-needed")
FAST_MODEL = os.getenv("FAST_MODEL", "")

MODE = "script"
DEBUG = os.getenv("DEBUG") == "1"
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))
TIMEOUT_SEC = 30

PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = PROJECT_ROOT / "logs"
SCREENSHOT_DIR = PROJECT_ROOT / "screenshots"
VOICE_DEFAULT_ENABLED = os.getenv("VOICE", "1") == "1"
VOICE_ENGINE = os.getenv("VOICE_ENGINE", "whisper").lower()
VOSK_MODEL_SIZE = os.getenv("VOSK_MODEL_SIZE", "full").lower()
_VOSK_MODEL_NAME = "vosk-model-small-ru-0.22" if VOSK_MODEL_SIZE == "small" else "vosk-model-ru-0.22"
VOSK_MODEL_DIR = Path(os.getenv("VOSK_MODEL_DIR", PROJECT_ROOT / "models" / _VOSK_MODEL_NAME))
WHISPER_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small").lower()
WHISPER_MODEL_NAME = os.getenv(
    "WHISPER_MODEL_NAME",
    "small" if WHISPER_MODEL_SIZE == "small" else "base",
)
WHISPER_DEVICE = os.getenv("WHISPER_DEVICE", "cuda").lower()
VOICE_SAMPLE_RATE = int(os.getenv("VOICE_SAMPLE_RATE", "16000"))
VOICE_DEVICE = os.getenv("VOICE_DEVICE")
VOICE_NAME = os.getenv("VOICE_NAME", "Microsoft Dmitry")
VOICE_RATE = int(os.getenv("VOICE_RATE", "2"))
VOICE_VOLUME = int(os.getenv("VOICE_VOLUME", "100"))

LOG_DIR.mkdir(parents=True, exist_ok=True)
SCREENSHOT_DIR.mkdir(parents=True, exist_ok=True)
