from __future__ import annotations

"""
wake_word_console.py

Единая консоль для wake-word:
- запись POS/NEG (короткие клипы)
- очистка мусора (auto-clean)
- тест последнего клипа
- полный тест по датасету (метрики + примеры + worst)
- быстрый тест (FAST TEST)
- sweep threshold (подбор порога)
- LIVE test (реальный микрофон)
- TURBO POS/NEG: длинная запись -> авто-нарезка по громкости -> сразу тест
- тренировка (scripts/train_real_wakeword.py) с сохранением настроек
- все настройки в одном месте + читаем/пишем в config.yaml

Главное:
✅ окно анализа (total_sec) везде одинаковое — как в тренировке
✅ countdown=0 реально без отсчёта (ничего не печатает)
✅ пункт 8 при входе берёт значения из config.yaml (threshold/min_rms/total_sec + training settings)
"""

import math
import random
import subprocess
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

try:
    import yaml
except Exception:
    yaml = None

try:
    import sounddevice as sd
except Exception:
    sd = None

try:
    from openwakeword.model import Model
except Exception:
    Model = None


# =========================
# Paths
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_POS_DIR = BASE_DIR / "data" / "positive"
DATA_NEG_DIR = BASE_DIR / "data" / "negative"
DATA_HARD_POS_DIR = BASE_DIR / "data" / "hard_positive"
DATA_HARD_NEG_DIR = BASE_DIR / "data" / "hard_negative"

MODELS_DIR = BASE_DIR / "models"
DEFAULT_MODEL_PATH = MODELS_DIR / "agent.onnx"
LAST_RECORDED_TXT = BASE_DIR / "data" / ".last_recorded.txt"

DEFAULT_AGENT_CONFIG = None
if (BASE_DIR / "voice_agent" / "config.yaml").exists():
    DEFAULT_AGENT_CONFIG = BASE_DIR / "voice_agent" / "config.yaml"
elif (BASE_DIR / "config.yaml").exists():
    DEFAULT_AGENT_CONFIG = BASE_DIR / "config.yaml"


# =========================
# YAML helpers
# =========================
def _load_yaml(path: Path) -> dict[str, Any]:
    if yaml is None or not path or not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _save_yaml(path: Path, data: dict[str, Any]) -> None:
    if yaml is None or not path:
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _ensure_cfg_sections(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Создаёт секции в YAML, если их нет.
    НИЧЕГО НЕ УДАЛЯЕТ.
    """
    cfg.setdefault("wake_word", {})
    cfg.setdefault("wake_word_train", {})
    cfg.setdefault("wake_word_recording", {})
    cfg.setdefault("wake_word_clean", {})
    cfg.setdefault("wake_word_fast_test", {})
    return cfg


def _resolve_model_path_from_yaml(cfg: dict[str, Any]) -> Path:
    ww = cfg.get("wake_word", {}) or {}
    mp = ww.get("model_paths", None)

    if isinstance(mp, list) and len(mp) > 0 and isinstance(mp[0], str) and mp[0].strip():
        candidate = Path(mp[0])
        if not candidate.is_absolute() and DEFAULT_AGENT_CONFIG is not None:
            candidate = DEFAULT_AGENT_CONFIG.parent / candidate
        return candidate

    # fallback
    return DEFAULT_MODEL_PATH


# =========================
# Audio / WAV
# =========================
def _read_wav_int16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
    return audio


def _write_wav_int16(path: Path, audio: np.ndarray, sr: int = 16000) -> None:
    audio = np.asarray(audio, dtype=np.int16)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())


def _rms(audio_int16: np.ndarray) -> float:
    if audio_int16.size == 0:
        return 0.0
    x = audio_int16.astype(np.float32) / 32768.0
    return float(math.sqrt(float(np.mean(x * x)) + 1e-12))


def _duration_s(audio_int16: np.ndarray, sr: int = 16000) -> float:
    if audio_int16.size == 0:
        return 0.0
    return float(audio_int16.size) / float(sr)


def _pad_or_trim(audio_int16: np.ndarray, total_len: int) -> np.ndarray:
    if audio_int16.shape[0] >= total_len:
        return audio_int16[:total_len].astype(np.int16, copy=False)
    out = np.zeros((total_len,), dtype=np.int16)
    out[: audio_int16.shape[0]] = audio_int16
    return out


# =========================
# Indexing (append mode)
# =========================
def _next_index(dir_path: Path, prefix: str) -> int:
    dir_path.mkdir(parents=True, exist_ok=True)
    best = 0
    for p in dir_path.glob(f"{prefix}_*.wav"):
        name = p.stem
        try:
            idx = int(name.split("_")[-1])
            best = max(best, idx)
        except Exception:
            continue
    return best + 1


def _save_last_recorded(path: Path) -> None:
    LAST_RECORDED_TXT.parent.mkdir(parents=True, exist_ok=True)
    LAST_RECORDED_TXT.write_text(str(path), encoding="utf-8")


def _load_last_recorded() -> Path | None:
    if not LAST_RECORDED_TXT.exists():
        return None
    p = Path(LAST_RECORDED_TXT.read_text(encoding="utf-8").strip())
    return p if p.exists() else None


# =========================
# openWakeWord scoring
# =========================
def _ensure_oww() -> None:
    if Model is None:
        print("[ERR] openwakeword не импортируется. Проверь: pip install openwakeword")
        raise SystemExit(1)


def _score_wav_fixed_window(
    model: Model,
    wav_path: Path,
    model_name: str,
    *,
    total_sec: float,
    sr: int = 16000,
) -> float:
    """
    score как в тренировке: фиксированное окно total_sec (pad/trim)
    """
    audio = _read_wav_int16(wav_path)
    total_len = int(sr * total_sec)
    audio = _pad_or_trim(audio, total_len)
    scores = model.predict(audio)
    return float(scores.get(model_name, 0.0))


# =========================
# Settings
# =========================
@dataclass
class TrainSettings:
    mode: str = "split"  # split/full
    epochs: int = 200
    batch: int = 256
    lr: float = 0.0005
    wd: float = 0.0001
    patience: int = 20
    rounds: int = 1
    mine_thr: float = 0.30
    feats_device: str = "cpu"  # cpu/cuda
    model_device: str = "cuda"  # cpu/cuda
    max_copy_neg: int = 500
    max_copy_pos: int = 150
    no_early_stop: bool = False


@dataclass
class ConsoleSettings:
    agent_name: str = "Бивис"
    sample_rate: int = 16000

    # окно анализа (важно одинаково как в трене)
    total_sec: float = 2.0

    # запись коротких клипов
    record_sec: float = 0.55
    countdown_sec: float = 2.0
    interval_sec: float = 0.35

    # порог громкости для сохранения
    min_rms_record: float = 0.008

    # wakeword
    model_path: Path = DEFAULT_MODEL_PATH
    model_name: str = "agent"
    threshold: float = 0.4

    # auto-clean
    min_duration_clean_s: float = 0.25
    min_rms_clean: float = 0.006

    # FAST TEST
    fast_test_pos_n: int = 80
    fast_test_neg_n: int = 200

    # Train (ВАЖНО: default_factory)
    train: TrainSettings = field(default_factory=TrainSettings)


# =========================
# Config: load/save unified
# =========================
def load_settings_from_agent_config(settings: ConsoleSettings) -> ConsoleSettings:
    """
    ✅ Загружает ВСЕ ключевые настройки из YAML, если YAML найден.
    НИЧЕГО НЕ ВЫРЕЗАЕМ: просто добавили ещё секции чтобы всё было согласовано из конфига.
    """
    if DEFAULT_AGENT_CONFIG is None:
        return settings

    cfg = _load_yaml(DEFAULT_AGENT_CONFIG)
    cfg = _ensure_cfg_sections(cfg)

    ww = cfg.get("wake_word", {}) or {}
    tr = cfg.get("wake_word_train", {}) or {}
    rec = cfg.get("wake_word_recording", {}) or {}
    cln = cfg.get("wake_word_clean", {}) or {}
    ft = cfg.get("wake_word_fast_test", {}) or {}

    # wake_word
    settings.threshold = float(ww.get("threshold", settings.threshold))
    settings.min_rms_record = float(ww.get("min_rms_record", settings.min_rms_record))
    settings.total_sec = float(ww.get("total_sec", settings.total_sec))
    settings.model_name = str(ww.get("model_name", settings.model_name))
    settings.agent_name = str(ww.get("agent_name", settings.agent_name))

    # model path
    try:
        settings.model_path = _resolve_model_path_from_yaml(cfg)
    except Exception:
        pass

    # recording
    settings.sample_rate = int(rec.get("sample_rate", settings.sample_rate))
    settings.record_sec = float(rec.get("record_sec", settings.record_sec))
    settings.countdown_sec = float(rec.get("countdown_sec", settings.countdown_sec))
    settings.interval_sec = float(rec.get("interval_sec", settings.interval_sec))

    # clean
    settings.min_duration_clean_s = float(cln.get("min_duration_clean_s", settings.min_duration_clean_s))
    settings.min_rms_clean = float(cln.get("min_rms_clean", settings.min_rms_clean))

    # fast test
    settings.fast_test_pos_n = int(ft.get("pos_n", settings.fast_test_pos_n))
    settings.fast_test_neg_n = int(ft.get("neg_n", settings.fast_test_neg_n))

    # train
    settings.train.mode = str(tr.get("mode", settings.train.mode))
    settings.train.epochs = int(tr.get("epochs", settings.train.epochs))
    settings.train.batch = int(tr.get("batch", settings.train.batch))
    settings.train.lr = float(tr.get("lr", settings.train.lr))
    settings.train.wd = float(tr.get("wd", settings.train.wd))
    settings.train.patience = int(tr.get("patience", settings.train.patience))
    settings.train.rounds = int(tr.get("rounds", settings.train.rounds))
    settings.train.mine_thr = float(tr.get("mine_thr", settings.train.mine_thr))
    settings.train.feats_device = str(tr.get("feats_device", settings.train.feats_device))
    settings.train.model_device = str(tr.get("model_device", settings.train.model_device))
    settings.train.max_copy_neg = int(tr.get("max_copy_neg", settings.train.max_copy_neg))
    settings.train.max_copy_pos = int(tr.get("max_copy_pos", settings.train.max_copy_pos))
    settings.train.no_early_stop = bool(tr.get("no_early_stop", settings.train.no_early_stop))

    return settings


def save_settings_to_agent_config(settings: ConsoleSettings) -> None:
    """
    ✅ Сохраняет ВСЕ настройки в YAML.
    Никаких шаблонов и дублей — один YAML = единый источник истины.
    """
    if DEFAULT_AGENT_CONFIG is None:
        print("[WARN] Не нашёл config.yaml — некуда сохранять.")
        return

    cfg = _load_yaml(DEFAULT_AGENT_CONFIG)
    cfg = _ensure_cfg_sections(cfg)

    # wake_word
    cfg["wake_word"]["threshold"] = float(settings.threshold)
    cfg["wake_word"]["min_rms_record"] = float(settings.min_rms_record)
    cfg["wake_word"]["total_sec"] = float(settings.total_sec)
    cfg["wake_word"]["model_name"] = str(settings.model_name)
    cfg["wake_word"]["agent_name"] = str(settings.agent_name)

    # сохраняем model_paths если есть
    # (не ломаем структуру: если уже был список — оставим, если нет — создадим)
    if "model_paths" not in cfg["wake_word"]:
        # стараемся сохранить относительный путь
        try:
            p = settings.model_path
            if DEFAULT_AGENT_CONFIG is not None and p.is_absolute():
                rel = p.relative_to(DEFAULT_AGENT_CONFIG.parent)
                cfg["wake_word"]["model_paths"] = [str(rel)]
            else:
                cfg["wake_word"]["model_paths"] = [str(p)]
        except Exception:
            cfg["wake_word"]["model_paths"] = [str(settings.model_path)]

    # recording
    cfg["wake_word_recording"]["sample_rate"] = int(settings.sample_rate)
    cfg["wake_word_recording"]["record_sec"] = float(settings.record_sec)
    cfg["wake_word_recording"]["countdown_sec"] = float(settings.countdown_sec)
    cfg["wake_word_recording"]["interval_sec"] = float(settings.interval_sec)

    # clean
    cfg["wake_word_clean"]["min_duration_clean_s"] = float(settings.min_duration_clean_s)
    cfg["wake_word_clean"]["min_rms_clean"] = float(settings.min_rms_clean)

    # fast test
    cfg["wake_word_fast_test"]["pos_n"] = int(settings.fast_test_pos_n)
    cfg["wake_word_fast_test"]["neg_n"] = int(settings.fast_test_neg_n)

    # train
    cfg["wake_word_train"]["mode"] = str(settings.train.mode)
    cfg["wake_word_train"]["epochs"] = int(settings.train.epochs)
    cfg["wake_word_train"]["batch"] = int(settings.train.batch)
    cfg["wake_word_train"]["lr"] = float(settings.train.lr)
    cfg["wake_word_train"]["wd"] = float(settings.train.wd)
    cfg["wake_word_train"]["patience"] = int(settings.train.patience)
    cfg["wake_word_train"]["rounds"] = int(settings.train.rounds)
    cfg["wake_word_train"]["mine_thr"] = float(settings.train.mine_thr)
    cfg["wake_word_train"]["feats_device"] = str(settings.train.feats_device)
    cfg["wake_word_train"]["model_device"] = str(settings.train.model_device)
    cfg["wake_word_train"]["max_copy_neg"] = int(settings.train.max_copy_neg)
    cfg["wake_word_train"]["max_copy_pos"] = int(settings.train.max_copy_pos)
    cfg["wake_word_train"]["no_early_stop"] = bool(settings.train.no_early_stop)

    _save_yaml(DEFAULT_AGENT_CONFIG, cfg)
    print(f"[OK] сохранено в {DEFAULT_AGENT_CONFIG}\n")


# =========================
# Metrics
# =========================
def _metrics_from_scores(pos_scores: list[float], neg_scores: list[float], thr: float) -> dict[str, float]:
    TP = sum(1 for s in pos_scores if s >= thr)
    FN = sum(1 for s in pos_scores if s < thr)
    TN = sum(1 for s in neg_scores if s < thr)
    FP = sum(1 for s in neg_scores if s >= thr)

    total = TP + TN + FP + FN
    acc = (TP + TN) / total if total else 0.0
    precision = TP / (TP + FP) if (TP + FP) else 0.0
    recall = TP / (TP + FN) if (TP + FN) else 0.0
    fpr = FP / (FP + TN) if (FP + TN) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return {
        "TP": float(TP),
        "FN": float(FN),
        "TN": float(TN),
        "FP": float(FP),
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "fpr": float(fpr),
        "f1": float(f1),
    }


# =========================
# Tests
# =========================
def dataset_test_human(settings: ConsoleSettings, *, random_show: int = 12, worst_show: int = 5) -> None:
    _ensure_oww()

    pos_wavs = sorted(DATA_POS_DIR.glob("*.wav"))
    neg_wavs = sorted(DATA_NEG_DIR.glob("*.wav"))

    if not pos_wavs and not neg_wavs:
        print("[ERR] Нет wav в data/positive и data/negative")
        return

    if not settings.model_path.exists():
        print(f"[ERR] Модель не найдена: {settings.model_path}")
        return

    print("\n[TEST] Загружаю wakeword-модель...")
    model = Model(wakeword_models=[str(settings.model_path)])
    print(f"[OK] model='{settings.model_name}' | threshold={settings.threshold:.3f}")
    print(f"[WINDOW] total_sec={settings.total_sec:.2f}s (как в тренировке)")
    print(f"[DATA] POS={len(pos_wavs)} | NEG={len(neg_wavs)}\n")

    pos_scored = [
        (p, _score_wav_fixed_window(model, p, settings.model_name, total_sec=settings.total_sec, sr=settings.sample_rate))
        for p in pos_wavs
    ]
    neg_scored = [
        (p, _score_wav_fixed_window(model, p, settings.model_name, total_sec=settings.total_sec, sr=settings.sample_rate))
        for p in neg_wavs
    ]

    pos_scores = [s for _, s in pos_scored]
    neg_scores = [s for _, s in neg_scored]
    m = _metrics_from_scores(pos_scores, neg_scores, settings.threshold)

    print("[ИТОГ (по датасету)]")
    print(f"✅ POS поймал: {int(m['TP'])}/{int(m['TP'] + m['FN'])} | ❌ пропустил: {int(m['FN'])}")
    print(f"✅ NEG игнор:  {int(m['TN'])}/{int(m['TN'] + m['FP'])} | ❌ ложняки:  {int(m['FP'])}")
    print(f"acc={m['acc']:.3f} | precision={m['precision']:.3f} | recall={m['recall']:.3f} | f1={m['f1']:.3f} | fpr={m['fpr']:.3f}\n")

    print("[ПРИМЕРЫ (случайные)]")
    if pos_scored:
        sample_pos = random.sample(pos_scored, k=min(random_show // 2, len(pos_scored)))
        print("POSITIVE:")
        for p, s in sample_pos:
            ok = s >= settings.threshold
            mark = "✅" if ok else "❌"
            msg = "OK" if ok else "FAIL"
            print(f"  {mark} {p.name}: score={s:.3f} -> {msg}")
        print()

    if neg_scored:
        sample_neg = random.sample(neg_scored, k=min(random_show // 2, len(neg_scored)))
        print("NEGATIVE:")
        for p, s in sample_neg:
            ok = s < settings.threshold
            mark = "✅" if ok else "❌"
            msg = "OK" if ok else "ЛОЖНЯК"
            print(f"  {mark} {p.name}: score={s:.3f} -> {msg}")
        print()

    worst_pos = sorted(pos_scored, key=lambda x: x[1])[:worst_show]
    worst_neg = sorted(neg_scored, key=lambda x: x[1], reverse=True)[:worst_show]

    if worst_pos:
        print("[САМЫЕ ПЛОХИЕ POSITIVE] (пропуски)")
        for p, s in worst_pos:
            print(f"  ❌ {p.name}: score={s:.3f}")
        print()

    if worst_neg:
        print("[САМЫЕ ОПАСНЫЕ NEGATIVE] (ложняки)")
        for p, s in worst_neg:
            print(f"  ❌ {p.name}: score={s:.3f}")
        print()


def fast_test(settings: ConsoleSettings) -> None:
    _ensure_oww()

    pos_wavs = sorted(DATA_POS_DIR.glob("*.wav"))
    neg_wavs = sorted(DATA_NEG_DIR.glob("*.wav"))
    if not pos_wavs or not neg_wavs:
        print("[ERR] Нужно и POS и NEG для FAST TEST.")
        return
    if not settings.model_path.exists():
        print(f"[ERR] Модель не найдена: {settings.model_path}")
        return

    pos_n = min(settings.fast_test_pos_n, len(pos_wavs))
    neg_n = min(settings.fast_test_neg_n, len(neg_wavs))
    pos_sel = random.sample(pos_wavs, k=pos_n)
    neg_sel = random.sample(neg_wavs, k=neg_n)

    print("\n[FAST TEST] Загружаю модель...")
    model = Model(wakeword_models=[str(settings.model_path)])
    print(f"[OK] thr={settings.threshold:.3f} | window={settings.total_sec:.2f}s | POS={pos_n} NEG={neg_n}")

    pos_scores = [
        _score_wav_fixed_window(model, p, settings.model_name, total_sec=settings.total_sec, sr=settings.sample_rate)
        for p in pos_sel
    ]
    neg_scores = [
        _score_wav_fixed_window(model, p, settings.model_name, total_sec=settings.total_sec, sr=settings.sample_rate)
        for p in neg_sel
    ]

    m = _metrics_from_scores(pos_scores, neg_scores, settings.threshold)
    print(f"acc={m['acc']:.3f} | precision={m['precision']:.3f} | recall={m['recall']:.3f} | f1={m['f1']:.3f} | fpr={m['fpr']:.3f}\n")


def test_last_recorded(settings: ConsoleSettings) -> None:
    _ensure_oww()
    p = _load_last_recorded()
    if p is None:
        print("[ERR] Не найден последний записанный wav")
        return
    if not settings.model_path.exists():
        print(f"[ERR] Модель не найдена: {settings.model_path}")
        return

    model = Model(wakeword_models=[str(settings.model_path)])
    s = _score_wav_fixed_window(model, p, settings.model_name, total_sec=settings.total_sec, sr=settings.sample_rate)

    is_pos = p.parent.name.lower() == "positive"
    ok = (s >= settings.threshold) if is_pos else (s < settings.threshold)
    mark = "✅" if ok else "❌"
    msg = "OK" if ok else "FAIL"

    print(f"\n[TEST LAST] {p.name}")
    print(f"window={settings.total_sec:.2f}s | score={s:.3f} | thr={settings.threshold:.3f} -> {mark} {msg}\n")


def sweep_threshold(settings: ConsoleSettings, step: float = 0.05) -> None:
    _ensure_oww()

    pos_wavs = sorted(DATA_POS_DIR.glob("*.wav"))
    neg_wavs = sorted(DATA_NEG_DIR.glob("*.wav"))

    if not pos_wavs:
        print("[ERR] Нет positive wav")
        return
    if not neg_wavs:
        print("[WARN] NEG почти нет. sweep будет кривой.")
        return
    if not settings.model_path.exists():
        print(f"[ERR] Модель не найдена: {settings.model_path}")
        return

    print("\n[SWEEP] Загружаю модель...")
    model = Model(wakeword_models=[str(settings.model_path)])
    print(f"[OK] model='{settings.model_name}' | window={settings.total_sec:.2f}s")

    pos_scores = [
        _score_wav_fixed_window(model, p, settings.model_name, total_sec=settings.total_sec, sr=settings.sample_rate)
        for p in pos_wavs
    ]
    neg_scores = [
        _score_wav_fixed_window(model, p, settings.model_name, total_sec=settings.total_sec, sr=settings.sample_rate)
        for p in neg_wavs
    ]

    thresholds = [round(float(x), 3) for x in np.arange(0.05, 0.96, step)]
    results: list[tuple[float, dict[str, float]]] = []
    for thr in thresholds:
        results.append((thr, _metrics_from_scores(pos_scores, neg_scores, thr)))

    best_thr = 0.5
    best_score = -1e9
    best_m = results[0][1]

    print("\n[SWEEP RESULT]")
    for thr, m in results:
        score = m["f1"] - 0.25 * m["fpr"]
        print(f"thr={thr:.2f} | recall={m['recall']:.3f} | FP={int(m['FP'])} | f1={m['f1']:.3f} | score={score:.3f}")
        if score > best_score:
            best_score = score
            best_thr = thr
            best_m = m

    print("\n✅ РЕКОМЕНДУЮ threshold:")
    print(f"thr={best_thr:.2f} | recall={best_m['recall']:.3f} | FP={int(best_m['FP'])} | f1={best_m['f1']:.3f}")
    print("Если ложнит — подними threshold. Если пропускает имя — снизь.\n")


# =========================
# Auto-clean
# =========================
def auto_clean(settings: ConsoleSettings) -> None:
    pos = sorted(DATA_POS_DIR.glob("*.wav"))
    neg = sorted(DATA_NEG_DIR.glob("*.wav"))
    removed = 0

    def check_one(p: Path) -> None:
        nonlocal removed
        try:
            a = _read_wav_int16(p)
        except Exception:
            p.unlink(missing_ok=True)
            removed += 1
            return

        dur = _duration_s(a, settings.sample_rate)
        rms = _rms(a)

        if dur < settings.min_duration_clean_s or rms < settings.min_rms_clean:
            p.unlink(missing_ok=True)
            removed += 1

    for p in pos:
        check_one(p)
    for p in neg:
        check_one(p)

    print(f"\n[CLEAN] Удалено клипов: {removed}")
    print(f"Правила: dur<{settings.min_duration_clean_s:.2f}s ИЛИ rms<{settings.min_rms_clean:.4f}\n")


# =========================
# Recording
# =========================
def _countdown(sec: float) -> None:
    if sec <= 0:
        return
    steps = int(math.ceil(sec))
    print("Приготовься...")
    for i in range(steps, 0, -1):
        print(f"... {i}")
        time.sleep(1.0)


def record_one_clip(settings: ConsoleSettings, *, is_positive: bool) -> Path | None:
    if sd is None:
        print("[ERR] sounddevice не установлен. pip install sounddevice")
        return None

    directory = DATA_POS_DIR if is_positive else DATA_NEG_DIR
    prefix = "positive" if is_positive else "negative"
    idx = _next_index(directory, prefix)
    out = directory / f"{prefix}_{idx:04d}.wav"

    _countdown(settings.countdown_sec)
    print("✅ ГОВОРИ СЕЙЧАС! ✅")

    n_samples = int(settings.sample_rate * settings.record_sec)
    audio = sd.rec(n_samples, samplerate=settings.sample_rate, channels=1, dtype="int16")
    sd.wait()
    audio = audio.reshape(-1)

    rms = _rms(audio)
    dur = _duration_s(audio, settings.sample_rate)
    print(f"[REC] rms={rms:.4f} dur={dur:.2f}s")

    if rms < settings.min_rms_record:
        print(f"[DROP] слишком тихо (min_rms_record={settings.min_rms_record:.4f}) -> НЕ сохранён\n")
        return None

    _write_wav_int16(out, audio, settings.sample_rate)
    _save_last_recorded(out)
    print(f"[SAVED] {out}")

    if settings.model_path.exists() and Model is not None:
        try:
            model = Model(wakeword_models=[str(settings.model_path)])
            s = _score_wav_fixed_window(model, out, settings.model_name, total_sec=settings.total_sec, sr=settings.sample_rate)
            ok = (s >= settings.threshold) if is_positive else (s < settings.threshold)
            mark = "✅" if ok else "❌"
            if is_positive:
                msg = "OK (имя поймано)" if ok else "FAIL (пропуск)"
                print(f"[TEST POS] score={s:.3f} thr={settings.threshold:.3f} -> {mark} {msg}\n")
            else:
                msg = "OK (не сработал)" if ok else "ЛОЖНЯК!"
                print(f"[TEST NEG] score={s:.3f} thr={settings.threshold:.3f} -> {mark} {msg}\n")
        except Exception as e:
            print(f"[TEST] не смог протестировать: {e}\n")
    else:
        print("[TEST] модель не найдена -> тест пропущен\n")

    return out


def ask_batch_record_params(settings: ConsoleSettings) -> None:
    print("\n[НАСТРОЙКА ЗАПИСИ ПАЧКИ]")
    print("Это применится ко ВСЕМ следующим клипам в этой пачке.")
    print(f"Сейчас: record_sec={settings.record_sec:.2f}s | countdown={settings.countdown_sec:.1f}s | interval={settings.interval_sec:.2f}s")
    print()

    try:
        raw1 = input(f"Длительность записи (сек)? [Enter={settings.record_sec:.2f}]: ").strip()
        if raw1:
            settings.record_sec = float(raw1)

        raw2 = input(f"Отсчёт перед записью (сек)? [Enter={settings.countdown_sec:.1f}] (можно 0): ").strip()
        if raw2:
            settings.countdown_sec = float(raw2)

        raw3 = input(f"Пауза между клипами (сек)? [Enter={settings.interval_sec:.2f}]: ").strip()
        if raw3:
            settings.interval_sec = float(raw3)

        print("\n[OK] Принято:")
        print(f"record_sec={settings.record_sec:.2f}s | countdown={settings.countdown_sec:.1f}s | interval={settings.interval_sec:.2f}s\n")

    except Exception as e:
        print(f"[ERR] Некорректный ввод: {e}")
        print("Оставляю старые значения.\n")


# =========================
# TURBO record: long -> slice -> save
# =========================
def _record_long_audio(settings: ConsoleSettings, sec: float) -> np.ndarray:
    if sd is None:
        raise RuntimeError("sounddevice missing")
    n_samples = int(settings.sample_rate * sec)
    print(f"[TURBO REC] запись {sec:.1f}s ... говори/шумите")
    audio = sd.rec(n_samples, samplerate=settings.sample_rate, channels=1, dtype="int16")
    sd.wait()
    return audio.reshape(-1)


def _slice_by_energy(
    audio: np.ndarray,
    *,
    sr: int,
    chunk_sec: float,
    hop_sec: float,
    rms_thr: float,
    min_keep_sec: float,
) -> List[Tuple[int, int]]:
    chunk = int(sr * chunk_sec)
    hop = int(sr * hop_sec)
    n = len(audio)

    active = []
    for i in range(0, max(1, n - chunk + 1), hop):
        seg = audio[i : i + chunk]
        r = _rms(seg)
        if r >= rms_thr:
            active.append((i, i + chunk))

    if not active:
        return []

    merged = [list(active[0])]
    for a, b in active[1:]:
        if a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])

    out = []
    min_len = int(sr * min_keep_sec)
    for a, b in merged:
        if (b - a) >= min_len:
            out.append((a, b))
    return out


def turbo_record_and_slice(settings: ConsoleSettings, *, is_positive: bool) -> None:
    if sd is None:
        print("[ERR] sounddevice не установлен.")
        return

    directory = DATA_POS_DIR if is_positive else DATA_NEG_DIR
    prefix = "positive" if is_positive else "negative"

    print("\n[TURBO MODE]")
    print("Записываем ОДИН длинный файл и автоматически режем на клипы по громкости.\n")

    try:
        sec = float(input("Длина длинной записи (сек) [30]: ").strip() or "30")
        rms_thr = float(input(f"Порог rms для нарезки [Enter={settings.min_rms_record:.4f}]: ").strip() or str(settings.min_rms_record))
        keep_sec = float(input("Минимальная длина кусочка (сек) [0.35]: ").strip() or "0.35")
        chunk_sec = float(input("Окно анализа (сек) [0.25]: ").strip() or "0.25")
        hop_sec = float(input("Шаг окна (сек) [0.08]: ").strip() or "0.08")
    except Exception as e:
        print(f"[ERR] ввод: {e}")
        return

    _countdown(settings.countdown_sec)
    audio = _record_long_audio(settings, sec)

    ranges = _slice_by_energy(
        audio,
        sr=settings.sample_rate,
        chunk_sec=chunk_sec,
        hop_sec=hop_sec,
        rms_thr=rms_thr,
        min_keep_sec=keep_sec,
    )

    if not ranges:
        print("[TURBO] Активных кусков не найдено (порог слишком высокий?)\n")
        return

    print(f"[TURBO] Найдено сегментов: {len(ranges)}")

    saved = 0
    for a, b in ranges:
        clip = audio[a:b]
        want_len = int(settings.sample_rate * settings.record_sec)
        clip = _pad_or_trim(clip, want_len)

        if _rms(clip) < settings.min_rms_record:
            continue

        idx = _next_index(directory, prefix)
        out = directory / f"{prefix}_{idx:04d}.wav"
        _write_wav_int16(out, clip, settings.sample_rate)
        saved += 1

    print(f"[TURBO] Сохранено клипов: {saved}\n")
    fast_test(settings)


# =========================
# LIVE mic test
# =========================
def live_mic_test(settings: ConsoleSettings) -> None:
    _ensure_oww()

    if sd is None:
        print("[ERR] sounddevice не установлен.")
        return
    if not settings.model_path.exists():
        print(f"[ERR] Модель не найдена: {settings.model_path}")
        return

    print("\n[LIVE TEST]")
    print(f"Имя: '{settings.agent_name}' | thr={settings.threshold:.3f} | window={settings.total_sec:.2f}s")
    print("Ctrl+C чтобы выйти.\n")

    model = Model(wakeword_models=[str(settings.model_path)])

    sr = int(settings.sample_rate)
    window_samples = int(sr * settings.total_sec)
    step_samples = int(sr * 0.10)
    cooldown_s = 1.2

    ring = np.zeros((0,), dtype=np.int16)
    last_print = 0.0
    cooldown_until = 0.0

    def append_ring_local(r: np.ndarray, x: np.ndarray) -> np.ndarray:
        r2 = np.concatenate([r, x])
        if r2.size > window_samples:
            r2 = r2[-window_samples:]
        return r2

    def callback(indata, frames, time_info, status) -> None:
        nonlocal ring
        x = indata[:, 0]
        if x.dtype != np.int16:
            x = (np.clip(x, -1, 1) * 32767).astype(np.int16)
        ring = append_ring_local(ring, x)

    try:
        with sd.InputStream(
            samplerate=sr,
            channels=1,
            dtype="int16",
            blocksize=step_samples,
            callback=callback,
        ):
            while True:
                time.sleep(0.01)
                now = time.time()
                if now - last_print < 0.10:
                    continue
                last_print = now

                if ring.size < window_samples:
                    print(f"\r[LIVE] буфер {ring.size}/{window_samples} ...", end="")
                    continue

                r = _rms(ring)
                if r < settings.min_rms_record:
                    print(f"\r[LIVE] тишина rms={r:.4f} (<{settings.min_rms_record:.4f})   ", end="")
                    continue

                if now < cooldown_until:
                    print(f"\r[LIVE] cooldown... rms={r:.4f}         ", end="")
                    continue

                scores = model.predict(ring)
                s = float(scores.get(settings.model_name, 0.0))

                if s >= settings.threshold:
                    cooldown_until = now + cooldown_s
                    print(f"\r✅ DETECTED '{settings.agent_name}' score={s:.3f} >= thr={settings.threshold:.3f}         ")
                else:
                    print(f"\r[LIVE] score={s:.3f} thr={settings.threshold:.3f} | rms={r:.4f}     ", end="")

    except KeyboardInterrupt:
        print("\n[LIVE] stopped.\n")


# =========================
# Training runner
# =========================
def train_real_model(settings: ConsoleSettings) -> None:
    script = BASE_DIR / "scripts" / "train_real_wakeword.py"
    if not script.exists():
        print(f"[TRAIN] Скрипт не найден: {script}")
        return

    # ✅ ЕДИНЫЙ ИСТОЧНИК ИСТИНЫ = voice_agent/config.yaml
    # ВАЖНО: НЕ передаём параметры через CLI, иначе они начнут расходиться с конфигом.
    # train_real_wakeword.py сам прочитает wake_word_train / wake_word из config.yaml.
    cmd = [sys.executable, str(script), "--save_to_config"]

    print("\n[TRAIN] запуск:")
    print(" ".join(cmd))
    try:
        subprocess.run(cmd, cwd=str(BASE_DIR), check=False)
    except Exception as e:
        print(f"[TRAIN ERR] {e}")


# =========================
# Explain
# =========================
def explain_threshold_and_rms(settings: ConsoleSettings) -> None:
    print("\n[ПОЯСНЕНИЕ]")
    print("threshold — порог СРАБАТЫВАНИЯ wake-word модели.")
    print("  score >= threshold  => агент считает: «услышал имя»")
    print("  score <  threshold  => агент считает: «имени нет»\n")
    print("min_rms_record — порог громкости для СОХРАНЕНИЯ клипа при записи датасета.")
    print("  rms < min_rms_record => клип не сохраняется (слишком тихо)\n")
    print("⚠️ Это разные вещи:")
    print("  • threshold = ложняки/пропуски wakeword")
    print("  • min_rms_record = сохранять ли запись вообще\n")


# =========================
# Settings menu
# =========================
def settings_menu(settings: ConsoleSettings) -> ConsoleSettings:
    print("\n[SETTINGS]")
    print(f"model_path: {settings.model_path}")
    print(f"model_name: {settings.model_name}")
    print(f"threshold: {settings.threshold:.3f}")
    print(f"min_rms_record: {settings.min_rms_record:.4f}")
    print(f"total_sec: {settings.total_sec:.2f}s")
    print(f"record_sec={settings.record_sec:.2f} | countdown={settings.countdown_sec:.1f} | interval={settings.interval_sec:.2f}\n")

    t = settings.train
    print("[TRAIN SETTINGS]")
    print(f"mode={t.mode} | epochs={t.epochs} | batch={t.batch} | lr={t.lr} | wd={t.wd}")
    print(f"patience={t.patience} | rounds={t.rounds} | mine_thr={t.mine_thr}")
    print(f"feats_device={t.feats_device} | model_device={t.model_device}")
    print(f"max_copy_neg={t.max_copy_neg} | max_copy_pos={t.max_copy_pos}")
    print(f"no_early_stop={t.no_early_stop}\n")

    try:
        raw_thr = input(f"Новый threshold? [Enter={settings.threshold:.3f}]: ").strip()
        if raw_thr:
            settings.threshold = float(raw_thr)

        raw_rms = input(f"Новый min_rms_record? [Enter={settings.min_rms_record:.4f}]: ").strip()
        if raw_rms:
            settings.min_rms_record = float(raw_rms)

        raw_win = input(f"Окно total_sec ? [Enter={settings.total_sec:.2f}]: ").strip()
        if raw_win:
            settings.total_sec = float(raw_win)

        raw_save = input("Сохранить ВСЕ настройки в config.yaml? [y/N]: ").strip().lower()
        if raw_save == "y":
            save_settings_to_agent_config(settings)

    except Exception as e:
        print(f"[ERR] {e}")

    print()
    return settings


# =========================
# Menu
# =========================
MENU = """
Wake-word console (реальные клипы + тесты)
1) Записать POSITIVE (имя «Бивис») + сразу OK/FAIL
2) Записать NEGATIVE (НЕ «Бивис») + сразу OK/FAIL
3) Очистить пустые клипы (auto-clean) из positive/negative
4) Тест: последний записанный wav (PASS/FAIL)
5) Тест: полный датасет positive/negative (метрики + примеры + worst)
6) Подбор threshold (sweep) по датасету + рекомендация
7) Обучить модель (train_real_wakeword.py) + hard-mining rounds
8) Настройки (берёт из config при входе) + сохранить все настройки
9) Пояснение: threshold vs min_rms
10) LIVE test (микрофон в реальном времени)
11) TURBO POSITIVE (длинная запись -> авто-нарезка -> быстрый тест)
12) TURBO NEGATIVE (длинная запись -> авто-нарезка -> быстрый тест)
13) FAST TEST (быстрый срез качества)
0) Выход
> """


def main() -> None:
    settings = ConsoleSettings()
    settings = load_settings_from_agent_config(settings)

    DATA_POS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_NEG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_HARD_POS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_HARD_NEG_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    while True:
        try:
            choice = input(MENU).strip()
        except KeyboardInterrupt:
            print("\n[EXIT]")
            break

        if choice == "0":
            print("[EXIT]")
            break

        elif choice == "1":
            ask_batch_record_params(settings)
            raw = input("Сколько POS записей сделать? [20]: ").strip()
            n = int(raw) if raw else 20

            saved = 0
            for i in range(n):
                p = record_one_clip(settings, is_positive=True)
                if p is not None:
                    saved += 1
                if i != n - 1:
                    time.sleep(max(0.0, settings.interval_sec))
            print(f"[POSITIVE] сохранено клипов: {saved}/{n}\n")

        elif choice == "2":
            ask_batch_record_params(settings)
            raw = input("Сколько NEG записей сделать? [20]: ").strip()
            n = int(raw) if raw else 20

            saved = 0
            for i in range(n):
                p = record_one_clip(settings, is_positive=False)
                if p is not None:
                    saved += 1
                if i != n - 1:
                    time.sleep(max(0.0, settings.interval_sec))
            print(f"[NEGATIVE] сохранено клипов: {saved}/{n}\n")

        elif choice == "3":
            auto_clean(settings)

        elif choice == "4":
            test_last_recorded(settings)

        elif choice == "5":
            dataset_test_human(settings, random_show=12, worst_show=5)

        elif choice == "6":
            sweep_threshold(settings, step=0.05)

        elif choice == "7":
            # ✅ чтобы тренировка 100% брала то что в yaml — сохраняем перед запуском
            save_settings_to_agent_config(settings)
            train_real_model(settings)

        elif choice == "8":
            # ✅ перечитываем конфиг перед меню, чтобы ты мог менять YAML руками тоже
            settings = load_settings_from_agent_config(settings)
            settings = settings_menu(settings)

        elif choice == "9":
            explain_threshold_and_rms(settings)

        elif choice == "10":
            live_mic_test(settings)

        elif choice == "11":
            turbo_record_and_slice(settings, is_positive=True)

        elif choice == "12":
            turbo_record_and_slice(settings, is_positive=False)

        elif choice == "13":
            fast_test(settings)

        else:
            print("[ERR] Неверный пункт меню.\n")


if __name__ == "__main__":
    main()
