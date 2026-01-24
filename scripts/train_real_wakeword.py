from __future__ import annotations

"""
train_real_wakeword.py

Обучение wake-word модели на реальных записях (data/positive, data/negative).

Фишки:
- GPU для PyTorch (если есть CUDA)
- Фичи (AudioFeatures) по умолчанию считаем на CPU (стабильно), но можно попробовать CUDA
- Режимы тренировок:
    --train_regime current : честный split 80/20 (train/val)
    --train_regime full    : почти весь датасет 90/10 (train/val) (честный контроль)
  (Совместимость со старым --mode split/full сохранена)
- Hard-mining по кругу:
    --rounds N : после каждого раунда находим hard_negative / hard_positive по всему датасету
- Offline-aug embeddings для TRAIN (важно! теперь реально используется в обучении)
- Кэширование/предподсчёт эмбеддингов: один раз считаем embeddings -> дальше быстро
- Экспорт:
    models/agent.onnx
    models/agent.pt
    models/agent.threshold.txt

Важно для твоего кейса:
- Если ложные срабатывания ~98% даже при thr=0.9, значит модель почти всегда даёт score~1 на NEG.
  Тогда "порогом" не лечится — нужно:
  ✅ сильнее штрафовать NEG в loss (anti-FP)
  ✅ hard-negative oversample
  ✅ подбирать thr с сильным штрафом FPR
"""

import argparse
import math
import os
import random
import sys
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from openwakeword.utils import AudioFeatures

try:
    import yaml
except Exception:
    yaml = None


# =========================
# Constants / Defaults
# =========================
SR = 16000
DEFAULT_TOTAL_SEC = 2.0
RNG_SEED = 42
ANSI_RESET = "\033[0m"
ANSI_RED = "\033[31m"


def _warn_red(text: str) -> str:
    return f"{ANSI_RED}{text}{ANSI_RESET}"


# =========================
# Config (shared with console)
# =========================
BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_AGENT_CONFIG = None
if (BASE_DIR / "voice_agent" / "config.yaml").exists():
    DEFAULT_AGENT_CONFIG = BASE_DIR / "voice_agent" / "config.yaml"
elif (BASE_DIR / "config.yaml").exists():
    DEFAULT_AGENT_CONFIG = BASE_DIR / "config.yaml"


def _load_yaml(path: Path) -> dict:
    if yaml is None or (not path) or (not path.exists()):
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _save_yaml(path: Path, data: dict) -> None:
    if yaml is None or (not path):
        return
    path.write_text(
        yaml.safe_dump(data, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def load_train_settings_from_config() -> dict:
    """
    Берём дефолты из voice_agent/config.yaml (если есть).
    Ничего не ломаем, если секций нет.
    """
    cfg = _load_yaml(DEFAULT_AGENT_CONFIG) if DEFAULT_AGENT_CONFIG else {}
    ww = cfg.get("wake_word", {}) or {}
    tr = cfg.get("wake_word_train", {}) or {}

    # defaults под анти-ложняк
    return {
        "threshold": float(ww.get("threshold", 0.40)),
        "min_rms_record": float(ww.get("min_rms_record", 0.008)),
        "total_sec": float(ww.get("total_sec", DEFAULT_TOTAL_SEC)),

        "mode": str(tr.get("mode", "split")),  # split/full (legacy)
        "epochs": int(tr.get("epochs", 200)),
        "batch": int(tr.get("batch", 128)),
        "lr": float(tr.get("lr", 5e-4)),
        "wd": float(tr.get("wd", 1e-4)),
        "patience": int(tr.get("patience", 25)),
        "rounds": int(tr.get("rounds", 3)),
        "mine_thr": float(tr.get("mine_thr", 0.75)),
        "max_copy_neg": int(tr.get("max_copy_neg", 800)),
        "max_copy_pos": int(tr.get("max_copy_pos", 200)),
        "layer": int(tr.get("layer", 128)),
        "feats_device": str(tr.get("feats_device", "cpu")),
        "model_device": str(tr.get("model_device", "cuda")),

        # Анти-FP параметры
        "neg_weight": float(tr.get("neg_weight", 6.0)),  # штраф NEG сильнее
        "pos_weight": float(tr.get("pos_weight", 1.0)),  # POS без усиления
        "fpr_penalty": float(tr.get("fpr_penalty", 3.0)),  # подбор thr с жёстким штрафом FPR

        "aug": (tr.get("aug", {}) or {}),
    }


def save_train_settings_to_config(settings: dict) -> None:
    if DEFAULT_AGENT_CONFIG is None:
        return

    cfg = _load_yaml(DEFAULT_AGENT_CONFIG)
    cfg.setdefault("wake_word", {})
    cfg.setdefault("wake_word_train", {})

    cfg["wake_word"]["threshold"] = float(settings.get("threshold", 0.40))
    cfg["wake_word"]["min_rms_record"] = float(settings.get("min_rms_record", 0.008))
    cfg["wake_word"]["total_sec"] = float(settings.get("total_sec", DEFAULT_TOTAL_SEC))

    cfg["wake_word_train"]["mode"] = str(settings.get("mode", "split"))
    cfg["wake_word_train"]["epochs"] = int(settings.get("epochs", 200))
    cfg["wake_word_train"]["batch"] = int(settings.get("batch", 128))
    cfg["wake_word_train"]["lr"] = float(settings.get("lr", 5e-4))
    cfg["wake_word_train"]["wd"] = float(settings.get("wd", 1e-4))
    cfg["wake_word_train"]["patience"] = int(settings.get("patience", 25))
    cfg["wake_word_train"]["rounds"] = int(settings.get("rounds", 3))
    cfg["wake_word_train"]["mine_thr"] = float(settings.get("mine_thr", 0.75))
    cfg["wake_word_train"]["max_copy_neg"] = int(settings.get("max_copy_neg", 800))
    cfg["wake_word_train"]["max_copy_pos"] = int(settings.get("max_copy_pos", 200))
    cfg["wake_word_train"]["layer"] = int(settings.get("layer", 128))
    cfg["wake_word_train"]["feats_device"] = str(settings.get("feats_device", "cpu"))
    cfg["wake_word_train"]["model_device"] = str(settings.get("model_device", "cuda"))

    # anti-FP
    cfg["wake_word_train"]["neg_weight"] = float(settings.get("neg_weight", 6.0))
    cfg["wake_word_train"]["pos_weight"] = float(settings.get("pos_weight", 1.0))
    cfg["wake_word_train"]["fpr_penalty"] = float(settings.get("fpr_penalty", 3.0))

    # augmentation settings (optional)
    if isinstance(settings.get("aug"), dict):
        cfg["wake_word_train"]["aug"] = settings["aug"]

    _save_yaml(DEFAULT_AGENT_CONFIG, cfg)


# =========================
# WAV helpers
# =========================
def read_wav_mono_16k(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)

    if sampwidth != 2:
        raise ValueError(f"{path}: expected 16-bit PCM (sampwidth=2), got {sampwidth}")
    if sr != SR:
        raise ValueError(f"{path}: expected {SR} Hz, got {sr}. Приведи wav к 16kHz.")

    audio = np.frombuffer(raw, dtype=np.int16)

    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1).astype(np.int16)

    return audio


def pad_or_trim(audio: np.ndarray, total_len: int) -> np.ndarray:
    if len(audio) >= total_len:
        return audio[:total_len].astype(np.int16, copy=False)
    out = np.zeros(total_len, dtype=np.int16)
    out[: len(audio)] = audio
    return out


# =========================
# Model (LOGITS)
# =========================
class DNNWakeword(nn.Module):
    """
    Последний слой = LOGIT (без Sigmoid).
    Потом используем BCEWithLogitsLoss.
    """

    def __init__(self, input_shape: tuple[int, int], layer_size: int = 128):
        super().__init__()
        n_in = input_shape[0] * input_shape[1]
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(n_in, layer_size),
            nn.LayerNorm(layer_size),
            nn.ReLU(),
            nn.Dropout(p=0.15),
            nn.Linear(layer_size, layer_size),
            nn.LayerNorm(layer_size),
            nn.ReLU(),
            nn.Dropout(p=0.10),
            nn.Linear(layer_size, 1),  # LOGIT
        )

    def forward(self, x):
        return self.net(x)


# =========================
# Metrics
# =========================
@dataclass
class Metrics:
    TP: int
    FN: int
    TN: int
    FP: int
    acc: float
    precision: float
    recall: float
    f1: float
    fpr: float


def compute_metrics(probs: torch.Tensor, y: torch.Tensor, thr: float) -> Metrics:
    pred = (probs >= thr).float()

    tp = int(((pred == 1) & (y == 1)).sum().item())
    fn = int(((pred == 0) & (y == 1)).sum().item())
    tn = int(((pred == 0) & (y == 0)).sum().item())
    fp = int(((pred == 1) & (y == 0)).sum().item())

    total = tp + tn + fp + fn
    acc = (tp + tn) / total if total else 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fpr = fp / (fp + tn) if (fp + tn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0

    return Metrics(tp, fn, tn, fp, acc, precision, recall, f1, fpr)


def find_best_threshold(
    probs: torch.Tensor,
    y: torch.Tensor,
    fpr_penalty: float = 3.0,
) -> Tuple[float, Metrics, float]:
    """
    Подбираем threshold по валу.
    score = f1 - fpr_penalty * fpr  (жёстко штрафуем ложняки)
    """
    best_thr = 0.5
    best_score = -1e9
    best_m = compute_metrics(probs, y, thr=0.5)

    for thr in np.linspace(0.05, 0.99, 95):
        m = compute_metrics(probs, y, thr=float(thr))
        score = m.f1 - float(fpr_penalty) * m.fpr
        if score > best_score:
            best_score = score
            best_thr = float(thr)
            best_m = m

    return best_thr, best_m, float(best_score)


def _print_score_stats(probs: torch.Tensor, y: torch.Tensor, name: str) -> None:
    probs_np = probs.view(-1).detach().cpu().numpy()
    y_np = y.view(-1).detach().cpu().numpy().astype(int)

    pos = probs_np[y_np == 1]
    neg = probs_np[y_np == 0]

    def fmt(a: np.ndarray) -> str:
        if a.size == 0:
            return "empty"
        return (
            f"mean={a.mean():.3f} | "
            f"p50={np.percentile(a, 50):.3f} "
            f"p90={np.percentile(a, 90):.3f} "
            f"p99={np.percentile(a, 99):.3f} "
            f"max={a.max():.3f}"
        )

    print(f"\n[SCORES {name}] POS: {fmt(pos)}")
    print(f"[SCORES {name}] NEG: {fmt(neg)}\n")


# =========================
# Dataset prep
# =========================
def load_dataset(
    repo: Path,
    *,
    extra_neg_dirs: List[Path] | None = None,
) -> Tuple[List[Path], List[Path], List[Path]]:
    pos_dir = repo / "data" / "positive"
    neg_dir = repo / "data" / "negative"

    pos = sorted(pos_dir.glob("*.wav"))
    neg = sorted(neg_dir.glob("*.wav"))

    if extra_neg_dirs:
        for d in extra_neg_dirs:
            neg.extend(sorted(d.glob("*.wav")))

    neg_extra: list[Path] = []
    if extra_neg_dirs:
        for d in extra_neg_dirs:
            neg_extra.extend(sorted(d.glob("*.wav")))

    return pos, neg, neg_extra


def split_items(items: List[Path], val_ratio: float, rng: random.Random) -> Tuple[List[Path], List[Path]]:
    items = items.copy()
    rng.shuffle(items)
    k = max(1, int(len(items) * val_ratio))
    return items[k:], items[:k]


def ensure_audiofeatures(feats_device: str, ncpu: int) -> AudioFeatures:
    """
    Некоторые сборки openwakeword умеют feats_device="cuda", некоторые — нет.
    Поэтому:
    - пробуем как просили
    - если не вышло, падаем на cpu и предупреждаем
    """
    feats_device = (feats_device or "cpu").lower().strip()
    if feats_device == "cuda" and not torch.cuda.is_available():
        print("[WARN] feats_device=cuda, но CUDA недоступна -> feats_device=cpu")
        feats_device = "cpu"

    try:
        F = AudioFeatures(device=feats_device, ncpu=ncpu)
        return F
    except Exception as e:
        print(f"[WARN] AudioFeatures(device='{feats_device}') не взлетел: {e}")
        print("[WARN] Переключаюсь на AudioFeatures(device='cpu')\n")
        return AudioFeatures(device="cpu", ncpu=ncpu)


def embed_all(F: AudioFeatures, wavs: List[Path], total_len: int, batch_size: int = 32) -> np.ndarray:
    """
    Считает embeddings для набора wav (CLEAN).
    Делает pad/trim до total_len строго.
    """
    pcm = np.stack([pad_or_trim(read_wav_mono_16k(p), total_len) for p in wavs], axis=0)
    emb = F.embed_clips(pcm, batch_size=batch_size)
    return emb


# =========================
# Offline Augmentations
# =========================
@dataclass
class AugmentConfig:
    enabled: bool = True
    pos_copies: int = 15         # включая "чистый" (1 = без доп копий)
    neg_copies: int = 1
    gain_db: float = 6.0         # случайный gain в диапазоне [-gain_db, +gain_db]
    speed_min: float = 0.92      # speed perturbation
    speed_max: float = 1.08
    mix_neg_prob: float = 0.35   # вероятность подмешать негатив как "фон" в POS
    mix_snr_db_min: float = 8.0
    mix_snr_db_max: float = 20.0
    trim_speech: bool = True     # убрать длинные пустыри вокруг речи (для POS особенно)
    speech_rms_thr: float = 0.006
    speech_pad_sec: float = 0.08 # сколько "воздуха" оставить до/после активной речи
    place_mode_train: str = "random"  # random|center|start
    place_mode_val: str = "center"


def _to_float32(pcm_i16: np.ndarray) -> np.ndarray:
    return (pcm_i16.astype(np.float32) / 32768.0).clip(-1.0, 1.0)


def _to_int16(x: np.ndarray) -> np.ndarray:
    return (np.clip(x, -1.0, 1.0) * 32767.0).astype(np.int16)


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(x), dtype=np.float32)) + 1e-12)


def _resample_linear(x: np.ndarray, speed: float) -> np.ndarray:
    # speed > 1.0 => быстрее (короче), speed < 1.0 => медленнее (длиннее)
    if speed <= 0:
        return x
    n = len(x)
    m = max(1, int(round(n / speed)))
    if m == n:
        return x
    xp = np.linspace(0.0, 1.0, num=n, dtype=np.float32)
    fp = x.astype(np.float32)
    xq = np.linspace(0.0, 1.0, num=m, dtype=np.float32)
    return np.interp(xq, xp, fp).astype(np.float32)


def _trim_speech_region(x: np.ndarray, sr: int, thr: float, pad_sec: float) -> np.ndarray:
    if len(x) < int(0.20 * sr):
        return x
    win = int(0.02 * sr)  # 20ms
    hop = int(0.01 * sr)  # 10ms
    if win <= 0 or hop <= 0:
        return x

    rms_vals = []
    for i in range(0, len(x) - win + 1, hop):
        seg = x[i : i + win]
        rms_vals.append(_rms(seg))
    if not rms_vals:
        return x

    rms_arr = np.asarray(rms_vals, dtype=np.float32)
    adaptive = float(np.percentile(rms_arr, 75) * 0.6)
    use_thr = max(float(thr), adaptive)
    idx = np.where(rms_arr >= use_thr)[0]
    if len(idx) == 0:
        return x

    start_frame = int(idx[0])
    end_frame = int(idx[-1])
    start = start_frame * hop
    end = end_frame * hop + win
    pad = int(pad_sec * sr)
    start = max(0, start - pad)
    end = min(len(x), end + pad)
    return x[start:end]


def _place_in_window(x: np.ndarray, total_len: int, rng: random.Random, mode: str) -> np.ndarray:
    # x - float32 audio (mono)
    if len(x) >= total_len:
        if mode == "random":
            start = rng.randint(0, max(0, len(x) - total_len))
        else:
            start = max(0, (len(x) - total_len) // 2)
        return x[start : start + total_len]

    out = np.zeros((total_len,), dtype=np.float32)
    free = total_len - len(x)
    if mode == "start":
        off = 0
    elif mode == "center":
        off = free // 2
    else:
        off = rng.randint(0, free)
    out[off : off + len(x)] = x
    return out


def _apply_gain(x: np.ndarray, rng: random.Random, gain_db: float) -> np.ndarray:
    if gain_db <= 0:
        return x
    db = rng.uniform(-gain_db, gain_db)
    g = float(10 ** (db / 20.0))
    return (x * g).astype(np.float32)


def _apply_speed(x: np.ndarray, rng: random.Random, smin: float, smax: float) -> np.ndarray:
    if smin <= 0 or smax <= 0 or smax <= smin:
        return x
    speed = rng.uniform(smin, smax)
    return _resample_linear(x, speed)


def _mix_background(signal: np.ndarray, bg: np.ndarray, rng: random.Random, snr_min: float, snr_max: float) -> np.ndarray:
    if len(bg) != len(signal):
        bg_i16 = pad_or_trim(_to_int16(bg), len(signal))
        bg = _to_float32(bg_i16)

    s_rms = _rms(signal)
    n_rms = _rms(bg)
    if s_rms < 1e-6 or n_rms < 1e-6:
        return signal

    snr_db = rng.uniform(snr_min, snr_max)
    target_n_rms = s_rms / (10 ** (snr_db / 20.0))
    scale = target_n_rms / (n_rms + 1e-9)
    mixed = signal + bg * float(scale)
    return mixed.astype(np.float32)


def make_augmented_window(
    path: Path,
    total_len: int,
    rng: random.Random,
    aug: AugmentConfig,
    label: int,
    neg_pool: List[Path] | None,
    augment: bool,
    place_mode: str,
) -> np.ndarray:
    pcm = read_wav_mono_16k(path)
    x = _to_float32(pcm)

    # для POS полезно вырезать пустыри вокруг речи
    if aug.trim_speech and label == 1:
        x = _trim_speech_region(x, SR, aug.speech_rms_thr, aug.speech_pad_sec)

    if augment and aug.enabled:
        x = _apply_speed(x, rng, aug.speed_min, aug.speed_max)
        x = _apply_gain(x, rng, aug.gain_db)

    xw = _place_in_window(x, total_len, rng, place_mode)

    # подмешиваем фон (только POS)
    if augment and aug.enabled and label == 1 and neg_pool and rng.random() < aug.mix_neg_prob:
        bg_path = neg_pool[rng.randint(0, len(neg_pool) - 1)]
        bg_pcm = read_wav_mono_16k(bg_path)
        bg = _to_float32(bg_pcm)
        bg = _place_in_window(bg, total_len, rng, "random")
        xw = _mix_background(xw, bg, rng, aug.mix_snr_db_min, aug.mix_snr_db_max)

    return _to_int16(xw)


def embed_augmented(
    F: AudioFeatures,
    wavs: List[Path],
    total_len: int,
    rng: random.Random,
    aug: AugmentConfig,
    label: int,
    copies: int,
    neg_pool: List[Path] | None,
    augment_train: bool,
    place_mode: str,
    batch_size: int = 32,
) -> np.ndarray:
    """
    Считает embeddings для wavs с offline-аугментациями (copies на файл).
    """
    if copies < 1:
        copies = 1

    pcm_list: List[np.ndarray] = []
    for p in wavs:
        # базовый (clean)
        pcm_list.append(
            make_augmented_window(
                p, total_len, rng, aug, label=label, neg_pool=neg_pool,
                augment=False, place_mode=place_mode
            )
        )
        # аугментированные копии
        for _ in range(copies - 1):
            pcm_list.append(
                make_augmented_window(
                    p, total_len, rng, aug, label=label, neg_pool=neg_pool,
                    augment=augment_train, place_mode=place_mode
                )
            )

    pcm = np.stack(pcm_list, axis=0)
    emb = F.embed_clips(pcm, batch_size=batch_size)
    return emb


# =========================
# Hard mining (по всему датасету)
# =========================
@dataclass
class HardMiningResult:
    hard_pos_idx: List[int]  # FN: pos где prob < mine_thr
    hard_neg_idx: List[int]  # FP: neg где prob >= mine_thr
    hard_pos_count: int
    hard_neg_count: int


def mine_hard_examples(
    probs_all: torch.Tensor,
    y_all: torch.Tensor,
    mine_thr: float,
) -> HardMiningResult:
    """
    hard_negative (FP): y=0, но prob >= mine_thr
    hard_positive (FN): y=1, но prob <  mine_thr
    """
    probs = probs_all.view(-1)
    y = y_all.view(-1)

    hard_neg = ((y == 0) & (probs >= mine_thr)).nonzero(as_tuple=False).view(-1).tolist()
    hard_pos = ((y == 1) & (probs < mine_thr)).nonzero(as_tuple=False).view(-1).tolist()

    return HardMiningResult(
        hard_pos_idx=hard_pos,
        hard_neg_idx=hard_neg,
        hard_pos_count=len(hard_pos),
        hard_neg_count=len(hard_neg),
    )


def copy_hard_files(
    repo: Path,
    all_wavs: List[Path],
    hard: HardMiningResult,
    max_copy_neg: int,
    max_copy_pos: int,
    *,
    hard_pos_dir: Path | None = None,
    hard_neg_dir: Path | None = None,
) -> Tuple[int, int]:
    """
    Физически копируем найденные hard примеры в:
      data/hard_positive
      data/hard_negative (или кастомные директории)
    """
    hard_pos_dir = hard_pos_dir or (repo / "data" / "hard_positive")
    hard_neg_dir = hard_neg_dir or (repo / "data" / "hard_negative")
    hard_pos_dir.mkdir(parents=True, exist_ok=True)
    hard_neg_dir.mkdir(parents=True, exist_ok=True)

    hard_pos_idx = hard.hard_pos_idx[: max_copy_pos]
    hard_neg_idx = hard.hard_neg_idx[: max_copy_neg]

    copied_pos = 0
    copied_neg = 0

    for i in hard_pos_idx:
        src = all_wavs[i]
        dst = hard_pos_dir / src.name
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
            copied_pos += 1

    for i in hard_neg_idx:
        src = all_wavs[i]
        dst = hard_neg_dir / src.name
        if not dst.exists():
            dst.write_bytes(src.read_bytes())
            copied_neg += 1

    return copied_pos, copied_neg


# =========================
# Training core
# =========================
def train_one_round(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    *,
    input_shape: Tuple[int, int],
    layer: int,
    lr: float,
    wd: float,
    batch: int,
    max_epochs: int,
    patience: int,
    model_device: torch.device,
    neg_weight: float,
    pos_weight: float,
    fpr_penalty: float,
    no_early_stop: bool,
) -> Tuple[Dict[str, torch.Tensor], float, Metrics, float]:
    """
    Возвращает:
      best_state_dict_cpu,
      best_thr,
      best_metrics,
      best_score
    """
    model = DNNWakeword(input_shape=input_shape, layer_size=layer).to(model_device)

    # Anti-FP: считаем BCE поэлементно и умножаем веса по классу
    bce = nn.BCEWithLogitsLoss(reduction="none")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.6, patience=4, verbose=False
    )

    best_score = -1e9
    best_state = None
    best_thr = 0.5
    best_m = None

    bad_epochs = 0
    N = int(X_train.shape[0])

    # веса как тензоры на нужном девайсе
    NEG_W = torch.tensor(float(neg_weight), dtype=torch.float32, device=model_device)
    POS_W = torch.tensor(float(pos_weight), dtype=torch.float32, device=model_device)

    for ep in range(1, max_epochs + 1):
        model.train()
        idx = torch.randperm(N)

        total_loss = 0.0
        for i in range(0, N, batch):
            b = idx[i : i + batch]
            xb = X_train[b].to(model_device, non_blocking=True)
            yb = y_train[b].to(model_device, non_blocking=True)

            logits = model(xb)
            loss_raw = bce(logits, yb)  # [B,1]

            # y=0 (NEG) штрафуем сильнее -> режем FP
            w = torch.where(yb > 0.5, POS_W, NEG_W)
            loss = (loss_raw * w).mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            total_loss += float(loss.item()) * int(len(b))

        total_loss /= max(1, N)

        model.eval()
        with torch.no_grad():
            val_logits = model(X_val.to(model_device, non_blocking=True))
            val_probs = torch.sigmoid(val_logits).detach().cpu()
            thr, m, score = find_best_threshold(
                val_probs,
                y_val.detach().cpu(),
                fpr_penalty=float(fpr_penalty),
            )

        scheduler.step(score)
        cur_lr = opt.param_groups[0]["lr"]

        if ep == 1 or ep % 5 == 0:
            print(
                f"[EP {ep:04d}] loss={total_loss:.4f} | "
                f"BEST(thr={thr:.2f}) acc={m.acc:.3f} prec={m.precision:.3f} rec={m.recall:.3f} "
                f"f1={m.f1:.3f} fpr={m.fpr:.3f} | score={score:.3f} | lr={cur_lr:.6f}"
            )

        if score > best_score + 1e-4:
            best_score = float(score)
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_thr = float(thr)
            best_m = m
            bad_epochs = 0
        else:
            bad_epochs += 1

        if (not no_early_stop) and bad_epochs >= patience:
            print(f"[EARLY STOP] нет улучшений {patience} эпох подряд.\n")
            break

    assert best_state is not None and best_m is not None
    return best_state, best_thr, best_m, float(best_score)


def export_model(repo: Path, state: Dict[str, torch.Tensor], input_shape: Tuple[int, int], layer: int) -> Tuple[Path, Path]:
    models_dir = repo / "models"
    models_dir.mkdir(exist_ok=True)
    pt_path = models_dir / "agent.pt"
    onnx_path = models_dir / "agent.onnx"

    torch.save(state, pt_path)

    model_cpu = DNNWakeword(input_shape=input_shape, layer_size=layer)
    model_cpu.load_state_dict(state)
    model_cpu.eval()

    class _SigmoidWrap(nn.Module):
        def __init__(self, base: nn.Module):
            super().__init__()
            self.base = base

        def forward(self, x):
            return torch.sigmoid(self.base(x))

    model_export = _SigmoidWrap(model_cpu)

    try:
        import onnx  # noqa: F401

        dummy = torch.rand(1, input_shape[0], input_shape[1])
        torch.onnx.export(
            model_export,
            dummy,
            str(onnx_path),
            opset_version=13,
            input_names=["features"],
            output_names=["agent"],
            dynamic_axes={"features": {0: "batch"}, "agent": {0: "batch"}},
        )
    except Exception as e:
        print("[WARN] ONNX export skipped:", e)
        print("[HINT] pip install onnx")
    return pt_path, onnx_path


# =========================
# Main / CLI
# =========================
def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    # legacy
    p.add_argument("--mode", choices=["split", "full"], default=None, help="split=80/20, full=90/10 (честный контроль)")

    # new clear regimes
    p.add_argument(
        "--train_regime",
        choices=["current", "full"],
        default=None,
        help="current=split 80/20, full=90/10 (почти весь датасет) + честный вал",
    )

    p.add_argument("--epochs", type=int, default=None, help="макс эпох в раунде")
    p.add_argument("--batch", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--wd", type=float, default=None, help="weight decay")
    p.add_argument("--patience", type=int, default=None)
    p.add_argument("--rounds", type=int, default=None, help="сколько кругов hard-mining (1=один тренинг)")
    p.add_argument("--mine_thr", type=float, default=None, help="порог для hard_negative/hard_positive")
    p.add_argument("--max_copy_neg", type=int, default=None)
    p.add_argument("--max_copy_pos", type=int, default=None)
    p.add_argument("--layer", type=int, default=None, help="размер скрытого слоя")
    p.add_argument("--total_sec", type=float, default=None)
    p.add_argument("--feats_device", choices=["cpu", "cuda"], default=None, help="где считать AudioFeatures")
    p.add_argument("--model_device", choices=["cpu", "cuda"], default=None, help="где обучать модель")

    # anti-FP knobs
    p.add_argument("--neg_weight", type=float, default=None, help="вес NEG в loss (больше = меньше FP)")
    p.add_argument("--pos_weight", type=float, default=None, help="вес POS в loss (обычно 1.0)")
    p.add_argument("--fpr_penalty", type=float, default=None, help="штраф FPR при подборе threshold (больше = меньше FP)")

    p.add_argument("--no_early_stop", action="store_true", help="не останавливать по patience, строго epochs")
    p.add_argument("--no_aug", action="store_true", help="отключить аугментации (raw клипы)")
    p.add_argument("--aug_active", type=int, default=1, help="включить аугментации (1/0)")
    p.add_argument("--pos_copies", type=int, default=15, help="сколько копий POS (вкл. clean)")
    p.add_argument("--neg_copies", type=int, default=1, help="сколько копий NEG (вкл. clean)")
    p.add_argument("--save_to_config", action="store_true", help="сохранить параметры обучения в config.yaml")
    return p


def main(argv: List[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    argv_list = argv if argv is not None else sys.argv[1:]

    # seeds
    random.seed(RNG_SEED)
    np.random.seed(RNG_SEED)
    torch.manual_seed(RNG_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RNG_SEED)

    cfg = load_train_settings_from_config()

    # режим: current/full -> split/full
    if args.train_regime is not None:
        mode = "split" if args.train_regime == "current" else "full"
    else:
        mode = args.mode or cfg["mode"]

    max_epochs = int(args.epochs if args.epochs is not None else cfg["epochs"])
    batch = int(args.batch if args.batch is not None else cfg["batch"])
    lr = float(args.lr if args.lr is not None else cfg["lr"])
    wd = float(args.wd) if args.wd is not None else float(cfg.get("wd", 1e-4))
    patience = int(args.patience if args.patience is not None else cfg["patience"])
    rounds = int(args.rounds if args.rounds is not None else cfg["rounds"])
    mine_thr = float(args.mine_thr if args.mine_thr is not None else cfg["mine_thr"])
    max_copy_neg = int(args.max_copy_neg if args.max_copy_neg is not None else cfg["max_copy_neg"])
    max_copy_pos = int(args.max_copy_pos if args.max_copy_pos is not None else cfg["max_copy_pos"])

    layer = int(args.layer) if args.layer is not None else int(cfg.get("layer", 128))
    feats_device = (args.feats_device or cfg["feats_device"]).lower()
    model_device_str = (args.model_device or cfg["model_device"]).lower()
    total_sec = float(args.total_sec if args.total_sec is not None else cfg["total_sec"])
    total_len = int(SR * total_sec)

    neg_weight = float(args.neg_weight) if args.neg_weight is not None else float(cfg.get("neg_weight", 6.0))
    pos_weight = float(args.pos_weight) if args.pos_weight is not None else float(cfg.get("pos_weight", 1.0))
    fpr_penalty = float(args.fpr_penalty) if args.fpr_penalty is not None else float(cfg.get("fpr_penalty", 3.0))

    # split train/val ratio
    if mode == "split":
        val_ratio = 0.20
    else:
        val_ratio = 0.10  # "почти весь датасет", но честный контроль

    # AudioFeatures
    ncpu = max(1, (os.cpu_count() or 4) // 2)
    F = ensure_audiofeatures(feats_device, ncpu)
    input_shape = F.get_embedding_shape(total_sec)
    print(f"[FEATS] input_shape={input_shape} | ncpu={ncpu} | AudioFeatures device={getattr(F, 'device', feats_device)}")
    model_input_sec = float(input_shape[0]) * 0.08
    if abs(model_input_sec - total_sec) > 0.1:
        print(
            _warn_red(
                f"[WARN] Config total_sec={total_sec:.2f} but model input sees only {model_input_sec:.2f} sec. "
                f"Update config to {model_input_sec:.2f}!"
            )
        )

    # augmentation config
    aug_cfg_raw = cfg.get("aug", {}) if isinstance(cfg.get("aug", {}), dict) else {}
    aug = AugmentConfig(
        enabled=bool(aug_cfg_raw.get("enabled", True)),
        pos_copies=int(aug_cfg_raw.get("pos_copies", 15)),
        neg_copies=int(aug_cfg_raw.get("neg_copies", 1)),
        gain_db=float(aug_cfg_raw.get("gain_db", 6.0)),
        speed_min=float(aug_cfg_raw.get("speed_min", 0.92)),
        speed_max=float(aug_cfg_raw.get("speed_max", 1.08)),
        mix_neg_prob=float(aug_cfg_raw.get("mix_neg_prob", 0.35)),
        mix_snr_db_min=float(aug_cfg_raw.get("mix_snr_db_min", 8.0)),
        mix_snr_db_max=float(aug_cfg_raw.get("mix_snr_db_max", 20.0)),
        trim_speech=bool(aug_cfg_raw.get("trim_speech", True)),
        speech_rms_thr=float(aug_cfg_raw.get("speech_rms_thr", 0.006)),
        speech_pad_sec=float(aug_cfg_raw.get("speech_pad_sec", 0.08)),
        place_mode_train=str(aug_cfg_raw.get("place_mode_train", "random")),
        place_mode_val=str(aug_cfg_raw.get("place_mode_val", "center")),
    )

    if "--pos_copies" in argv_list:
        aug.pos_copies = int(args.pos_copies)
    if "--neg_copies" in argv_list:
        aug.neg_copies = int(args.neg_copies)
    if "--aug_active" in argv_list:
        aug.enabled = bool(int(args.aug_active))

    if args.no_aug:
        aug.enabled = False
        aug.pos_copies = 1
        aug.neg_copies = 1

    print(f"[AUG] enabled={aug.enabled} | pos_copies={aug.pos_copies} | neg_copies={aug.neg_copies}")

    # devices
    if model_device_str == "cuda" and not torch.cuda.is_available():
        print("[WARN] model_device=cuda, но CUDA недоступна -> model_device=cpu")
        model_device_str = "cpu"
    model_device = torch.device(model_device_str)

    print(f"[TRAIN] device={model_device} mode={mode} epochs={max_epochs} batch={batch} lr={lr} wd={wd} patience={patience}")
    print(f"       neg_weight={neg_weight:.2f} pos_weight={pos_weight:.2f} | fpr_penalty={fpr_penalty:.2f}")
    print(f"       rounds={rounds} | mine_thr={mine_thr:.2f}")
    if model_device.type == "cuda":
        print(f"       GPU: {torch.cuda.get_device_name(0)}")
    print()

    best_state_overall = None
    best_thr_overall = 0.5
    best_score_overall = -1e9
    best_metrics_overall = None

    rng = random.Random(RNG_SEED)
    pos_all, neg_base, _ = load_dataset(BASE_DIR)
    if not pos_all or not neg_base:
        print(f"[ERR] Нет данных: positive={len(pos_all)} negative={len(neg_base)}")
        return 2

    train_pos, val_pos = split_items(pos_all, val_ratio, rng)
    train_neg_base, val_neg = split_items(neg_base, val_ratio, rng)

    val_set = set(val_pos + val_neg)
    train_set = set(train_pos + train_neg_base)
    overlap = train_set & val_set
    if overlap:
        print(_warn_red(f"[WARN] train/val пересечение: {len(overlap)} файлов"))
        raise AssertionError("train/val overlap detected")

    print(f"[DATA] positive={len(pos_all)} negative={len(neg_base)}")
    print(f"[SPLIT] mode={mode} val_ratio={val_ratio:.2f}")
    print(f"        pos train={len(train_pos)} val={len(val_pos)} | neg train={len(train_neg_base)} val={len(val_neg)}")

    # hard-mining pools
    hard_pos_idx_pool: List[int] = []
    hard_neg_idx_pool: List[int] = []
    hard_dirs: List[Path] = []

    for r in range(1, rounds + 1):
        print(f"\n========== ROUND {r}/{rounds} ==========")

        _, _, neg_extra = load_dataset(
            BASE_DIR,
            extra_neg_dirs=hard_dirs,
        )
        train_neg = train_neg_base + neg_extra

        train_set = set(train_pos + train_neg)
        overlap = train_set & val_set
        if overlap:
            print(_warn_red(f"[WARN] train/val пересечение: {len(overlap)} файлов"))
            raise AssertionError("train/val overlap detected")

        print(f"[ROUND DATA] train_pos={len(train_pos)} train_neg={len(train_neg)} val_pos={len(val_pos)} val_neg={len(val_neg)}")
        if neg_extra:
            print(f"[ROUND DATA] hard_neg_extra={len(neg_extra)}")

        # ===============================
        # PRECOMPUTE embeddings for TRAIN/VAL
        # ===============================
        print("[EMB] считаю embeddings для TRAIN (offline-aug)...")
        X_pos_tr = (
            embed_augmented(
                F, train_pos, total_len, rng, aug,
                label=1, copies=aug.pos_copies, neg_pool=train_neg,
                augment_train=True, place_mode=aug.place_mode_train, batch_size=32
            ) if train_pos else np.zeros((0, *input_shape), dtype=np.float32)
        )
        X_neg_tr = (
            embed_augmented(
                F, train_neg, total_len, rng, aug,
                label=0, copies=aug.neg_copies, neg_pool=None,
                augment_train=False, place_mode=aug.place_mode_train, batch_size=32
            ) if train_neg else np.zeros((0, *input_shape), dtype=np.float32)
        )

        print("[EMB] считаю embeddings для VAL (clean)...")
        X_pos_val = (
            embed_augmented(
                F, val_pos, total_len, rng, aug,
                label=1, copies=1, neg_pool=None,
                augment_train=False, place_mode=aug.place_mode_val, batch_size=32
            ) if val_pos else np.zeros((0, *input_shape), dtype=np.float32)
        )
        X_neg_val = (
            embed_augmented(
                F, val_neg, total_len, rng, aug,
                label=0, copies=1, neg_pool=None,
                augment_train=False, place_mode=aug.place_mode_val, batch_size=32
            ) if val_neg else np.zeros((0, *input_shape), dtype=np.float32)
        )

        # Torch tensors (CPU)
        X_train_base = torch.from_numpy(np.concatenate([X_pos_tr, X_neg_tr], axis=0)).float()
        y_train_base = torch.from_numpy(
            np.array([1] * len(X_pos_tr) + [0] * len(X_neg_tr), dtype=np.float32)
        ).float().unsqueeze(1)

        X_val = torch.from_numpy(np.concatenate([X_pos_val, X_neg_val], axis=0)).float()
        y_val = torch.from_numpy(
            np.array([1] * len(X_pos_val) + [0] * len(X_neg_val), dtype=np.float32)
        ).float().unsqueeze(1)

        # FULL embeddings for hard-mining (CLEAN)
        print("[EMB] считаю embeddings для TRAIN DATA (для hard-mining)...")
        all_wavs = train_pos + train_neg
        y_all_np = np.array([1] * len(train_pos) + [0] * len(train_neg), dtype=np.float32)
        y_all = torch.from_numpy(y_all_np).float().unsqueeze(1)

        X_all = embed_all(F, all_wavs, total_len, batch_size=32)
        X_all_t = torch.from_numpy(X_all).float()

        # TRAIN set: augmented base + hard oversample
        X_train = X_train_base
        y_train = y_train_base

        if hard_pos_idx_pool or hard_neg_idx_pool:
            hard_pos_idx_pool = hard_pos_idx_pool[: max_copy_pos]
            hard_neg_idx_pool = hard_neg_idx_pool[: max_copy_neg]

            X_hard_pos = X_all_t[hard_pos_idx_pool] if hard_pos_idx_pool else torch.zeros((0, *input_shape))
            y_hard_pos = y_all[hard_pos_idx_pool] if hard_pos_idx_pool else torch.zeros((0, 1))

            X_hard_neg = X_all_t[hard_neg_idx_pool] if hard_neg_idx_pool else torch.zeros((0, *input_shape))
            y_hard_neg = y_all[hard_neg_idx_pool] if hard_neg_idx_pool else torch.zeros((0, 1))

            # ВАЖНО: главная боль — ложные срабатывания -> усиливаем NEG очень сильно
            repeat_neg = 10
            repeat_pos = 2

            X_extra = torch.cat(
                [X_hard_pos.repeat((repeat_pos, 1, 1)), X_hard_neg.repeat((repeat_neg, 1, 1))],
                dim=0,
            )
            y_extra = torch.cat(
                [y_hard_pos.repeat((repeat_pos, 1)), y_hard_neg.repeat((repeat_neg, 1))],
                dim=0,
            )

            X_train = torch.cat([X_train, X_extra], dim=0)
            y_train = torch.cat([y_train, y_extra], dim=0)

            print(f"[HARD] добавлено: hard_pos={len(hard_pos_idx_pool)}x{repeat_pos}, hard_neg={len(hard_neg_idx_pool)}x{repeat_neg}")
            print(f"[HARD] итог train size: {len(X_train)}")

        best_state, best_thr, best_m, best_score = train_one_round(
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val,
            input_shape=input_shape,
            layer=layer,
            lr=lr,
            wd=wd,
            batch=batch,
            max_epochs=max_epochs,
            patience=patience,
            model_device=model_device,
            neg_weight=neg_weight,
            pos_weight=pos_weight,
            fpr_penalty=fpr_penalty,
            no_early_stop=args.no_early_stop,
        )

        if best_score > best_score_overall:
            best_score_overall = best_score
            best_state_overall = best_state
            best_thr_overall = best_thr
            best_metrics_overall = best_m

        print("\n[EVAL FULL] ищу hard_negative/hard_positive по ВСЕМ файлам...")
        model_cpu = DNNWakeword(input_shape=input_shape, layer_size=layer)
        model_cpu.load_state_dict(best_state)
        model_cpu.eval()

        with torch.no_grad():
            logits_all = model_cpu(X_all_t)
            probs_all = torch.sigmoid(logits_all)

        _print_score_stats(probs_all, y_all, "FULL")

        hard = mine_hard_examples(probs_all, y_all, mine_thr=mine_thr)

        print(f"[HARD FOUND] hard_negative(FP)={hard.hard_neg_count} | hard_positive(FN)={hard.hard_pos_count}")
        print(f"[HARD POOL] train_size={len(all_wavs)} | hard_pos={hard.hard_pos_count} hard_neg={hard.hard_neg_count}")

        # сортировки: NEG по убыванию score (самые опасные), POS по возрастанию (самые провальные)
        if hard.hard_neg_idx:
            hard.hard_neg_idx = sorted(hard.hard_neg_idx, key=lambda i: float(probs_all[i].item()), reverse=True)
        if hard.hard_pos_idx:
            hard.hard_pos_idx = sorted(hard.hard_pos_idx, key=lambda i: float(probs_all[i].item()))

        hard_neg_idx_pool = hard.hard_neg_idx[: max_copy_neg]
        hard_pos_idx_pool = hard.hard_pos_idx[: max_copy_pos]

        hard_paths = [all_wavs[i] for i in hard_neg_idx_pool + hard_pos_idx_pool]
        leaked = set(hard_paths) & val_set
        if leaked:
            print(_warn_red(f"[WARN] hard примеры из val: {len(leaked)} файлов"))
            raise AssertionError("hard mining leakage detected")

        hard_neg_dir = BASE_DIR / "data" / f"hard_neg_round_{r}"
        copied_pos, copied_neg = copy_hard_files(
            BASE_DIR,
            all_wavs,
            hard,
            max_copy_neg=max_copy_neg,
            max_copy_pos=max_copy_pos,
            hard_neg_dir=hard_neg_dir,
        )
        hard_dirs.append(hard_neg_dir)
        print(f"[HARD SAVE] copied hard_negative={copied_neg} -> {hard_neg_dir}")
        print(f"[HARD SAVE] copied hard_positive={copied_pos} (в data/hard_positive)")

        if hard.hard_neg_count == 0 and hard.hard_pos_count == 0:
            print("[STOP] hard примеры больше не находятся -> выходим раньше.\n")
            break

    assert best_state_overall is not None and best_metrics_overall is not None

    pt_path, onnx_path = export_model(BASE_DIR, best_state_overall, input_shape, layer)

    thr_path = (BASE_DIR / "models" / "agent.threshold.txt")
    thr_path.write_text(f"{best_thr_overall:.4f}\n", encoding="utf-8")

    print("\n[FINAL BEST]")
    m = best_metrics_overall
    print(f"best_threshold={best_thr_overall:.2f}")
    print(f"acc={m.acc:.3f} precision={m.precision:.3f} recall={m.recall:.3f} f1={m.f1:.3f} fpr={m.fpr:.3f}")
    print(f"Saved: {pt_path}")
    if onnx_path.exists():
        print(f"Saved: {onnx_path}")
    print(f"Saved: {thr_path} -> {best_thr_overall:.4f}\n")

    if args.save_to_config:
        save_train_settings_to_config(
            {
                "threshold": best_thr_overall,
                "min_rms_record": cfg.get("min_rms_record", 0.008),
                "total_sec": total_sec,
                "mode": mode,
                "epochs": max_epochs,
                "batch": batch,
                "lr": lr,
                "wd": wd,
                "patience": patience,
                "rounds": rounds,
                "mine_thr": mine_thr,
                "max_copy_neg": max_copy_neg,
                "max_copy_pos": max_copy_pos,
                "layer": layer,
                "neg_weight": neg_weight,
                "pos_weight": pos_weight,
                "fpr_penalty": fpr_penalty,
                "aug": {
                    "enabled": bool(aug.enabled),
                    "pos_copies": int(aug.pos_copies),
                    "neg_copies": int(aug.neg_copies),
                    "gain_db": float(aug.gain_db),
                    "speed_min": float(aug.speed_min),
                    "speed_max": float(aug.speed_max),
                    "mix_neg_prob": float(aug.mix_neg_prob),
                    "mix_snr_db_min": float(aug.mix_snr_db_min),
                    "mix_snr_db_max": float(aug.mix_snr_db_max),
                    "trim_speech": bool(aug.trim_speech),
                    "speech_rms_thr": float(aug.speech_rms_thr),
                    "speech_pad_sec": float(aug.speech_pad_sec),
                    "place_mode_train": str(aug.place_mode_train),
                    "place_mode_val": str(aug.place_mode_val),
                },
                "feats_device": feats_device,
                "model_device": model_device_str,
            }
        )
        print(f"[CFG] Сохранил настройки в {DEFAULT_AGENT_CONFIG}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
