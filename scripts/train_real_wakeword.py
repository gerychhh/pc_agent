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
import hashlib
import math
import os
import random
import sys
import time
import wave
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
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
MODEL_VERSION = "openwakeword_v1"


def _warn_red(text: str) -> str:
    return f"{ANSI_RED}{text}{ANSI_RESET}"


def _sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _safe_float(value: object, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _safe_int(value: object, default: int) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _parse_bool(value: object, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    s = str(value).strip().lower()
    if s in {"1", "true", "yes", "y", "on"}:
        return True
    if s in {"0", "false", "no", "n", "off"}:
        return False
    return default


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


def _select_window(pcm: np.ndarray, total_len: int, mode: str, seed: int) -> np.ndarray:
    if len(pcm) <= total_len:
        return pad_or_trim(pcm, total_len)
    if mode == "center":
        start = max(0, (len(pcm) - total_len) // 2)
    else:
        rng = random.Random(seed)
        start = rng.randint(0, max(0, len(pcm) - total_len))
    return pcm[start : start + total_len].astype(np.int16, copy=False)


def _cache_key(item: CacheItem, feature_id: str) -> str:
    p = item.path.resolve()
    try:
        stat = p.stat()
        size = stat.st_size
        mtime = stat.st_mtime
    except Exception:
        size = 0
        mtime = 0.0
    material = "|".join(
        [
            str(p),
            str(size),
            str(mtime),
            str(item.sr),
            f"{item.total_sec:.3f}",
            item.window_mode,
            feature_id,
            MODEL_VERSION,
        ]
    )
    return _sha1(material)


def _cache_path(cache_dir: Path, item: CacheItem, key: str) -> Path:
    split_dir = cache_dir / item.split
    split_dir.mkdir(parents=True, exist_ok=True)
    return split_dir / f"{key}.pt"


def _cache_worker(payload: dict) -> tuple[str, bool, float]:
    start = time.perf_counter()
    item = payload["item"]
    cache_path = Path(payload["cache_path"])
    total_len = int(payload["sr"] * payload["total_sec"])
    feature_id = payload["feature_id"]
    device = payload["device"]

    if cache_path.exists():
        return str(cache_path), True, time.perf_counter() - start

    try:
        pcm = read_wav_mono_16k(Path(item["path"]))
        seed = int(_sha1(str(item["path"]) + feature_id + MODEL_VERSION), 16) % (2**31)
        window = _select_window(pcm, total_len, item["window_mode"], seed=seed)
        F = AudioFeatures(device=device, ncpu=1)
        emb = F.embed_clips(window[None, :], batch_size=1)[0]
        torch.save(
            {
                "embedding": torch.from_numpy(emb.astype(np.float32)),
                "label": int(item["label"]),
                "meta": {
                    "path": item["path"],
                    "mtime": item["mtime"],
                    "total_sec": item["total_sec"],
                    "sr": item["sr"],
                    "window_mode": item["window_mode"],
                    "feature_id": feature_id,
                    "model_version": MODEL_VERSION,
                },
            },
            cache_path,
        )
        return str(cache_path), False, time.perf_counter() - start
    except Exception:
        return str(cache_path), False, time.perf_counter() - start


def prune_cache(cache_dir: Path, max_gb: float) -> None:
    if max_gb <= 0:
        return
    if not cache_dir.exists():
        return
    files = sorted(cache_dir.rglob("*.pt"), key=lambda p: p.stat().st_mtime)
    total_bytes = sum(p.stat().st_size for p in files)
    limit = int(max_gb * (1024 ** 3))
    if total_bytes <= limit:
        return
    for p in files:
        try:
            size = p.stat().st_size
            p.unlink(missing_ok=True)
            total_bytes -= size
            if total_bytes <= limit:
                break
        except Exception:
            continue


def build_embedding_cache(
    items: List[CacheItem],
    cache_dir: Path,
    *,
    cache_device: str,
    num_workers: int,
    feature_id: str,
    total_sec: float,
    sr: int,
) -> tuple[int, int, float]:
    hits = 0
    misses = 0
    compute_time = 0.0
    if not items:
        return hits, misses, compute_time

    payloads = []
    for item in items:
        key = _cache_key(item, feature_id)
        cache_path = _cache_path(cache_dir, item, key)
        if cache_path.exists():
            hits += 1
            continue
        payloads.append(
            {
                "item": {
                    "path": str(item.path),
                    "label": int(item.label),
                    "sr": int(item.sr),
                    "total_sec": float(item.total_sec),
                    "window_mode": str(item.window_mode),
                    "split": str(item.split),
                    "mtime": item.path.stat().st_mtime if item.path.exists() else 0.0,
                },
                "cache_path": str(cache_path),
                "feature_id": feature_id,
                "device": cache_device,
                "sr": sr,
                "total_sec": total_sec,
            }
        )

    if not payloads:
        return hits, misses, compute_time

    with ProcessPoolExecutor(max_workers=max(1, num_workers)) as ex:
        futures = [ex.submit(_cache_worker, payload) for payload in payloads]
        for fut in as_completed(futures):
            _, hit, t = fut.result()
            if hit:
                hits += 1
            else:
                misses += 1
                compute_time += t

    return hits, misses, compute_time


class EmbeddingDataset(Dataset):
    def __init__(self, items: List[CacheItem], cache_dir: Path, *, train: bool, feature_id: str) -> None:
        self.items = items
        self.cache_dir = cache_dir
        self.train = train
        self.feature_id = feature_id

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.items[idx]
        key = _cache_key(item, self.feature_id)
        path = _cache_path(self.cache_dir, item, key)
        try:
            data = torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            data = torch.load(path, map_location="cpu")
        emb = data["embedding"].float()
        if self.train:
            emb = self._augment(emb)
        label = torch.tensor([float(data["label"])], dtype=torch.float32)
        return emb, label

    @staticmethod
    def _augment(emb: torch.Tensor) -> torch.Tensor:
        noise = torch.randn_like(emb) * random.uniform(0.01, 0.03)
        scale = random.uniform(0.9, 1.1)
        dropout_p = random.uniform(0.05, 0.15)
        mask = torch.rand_like(emb) > dropout_p
        return (emb * scale + noise) * mask


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


@dataclass
class CacheItem:
    path: Path
    label: int
    sr: int
    total_sec: float
    window_mode: str
    split: str


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


def select_hard_by_scores(
    probs_all: torch.Tensor,
    y_all: torch.Tensor,
    *,
    k_pos: int,
    k_neg: int,
) -> HardMiningResult:
    probs = probs_all.view(-1)
    y = y_all.view(-1)

    neg_idx = (y == 0).nonzero(as_tuple=False).view(-1)
    pos_idx = (y == 1).nonzero(as_tuple=False).view(-1)

    hard_neg = []
    hard_pos = []
    if len(neg_idx) > 0 and k_neg > 0:
        neg_scores = probs[neg_idx]
        top_neg = torch.topk(neg_scores, k=min(k_neg, len(neg_scores)), largest=True).indices
        hard_neg = neg_idx[top_neg].tolist()
    if len(pos_idx) > 0 and k_pos > 0:
        pos_scores = probs[pos_idx]
        top_pos = torch.topk(pos_scores, k=min(k_pos, len(pos_scores)), largest=False).indices
        hard_pos = pos_idx[top_pos].tolist()

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


def train_one_round_loader(
    train_loader: DataLoader,
    val_loader: DataLoader,
    *,
    input_shape: Tuple[int, int],
    layer: int,
    lr: float,
    wd: float,
    max_epochs: int,
    patience: int,
    model_device: torch.device,
    neg_weight: float,
    pos_weight: float,
    fpr_penalty: float,
    no_early_stop: bool,
    amp: bool,
    grad_accum_steps: int,
) -> Tuple[Dict[str, torch.Tensor], float, Metrics, float, float]:
    model = DNNWakeword(input_shape=input_shape, layer_size=layer).to(model_device)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.6, patience=4, verbose=False
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    best_score = -1e9
    best_state = None
    best_thr = 0.5
    best_m = None
    bad_epochs = 0

    NEG_W = torch.tensor(float(neg_weight), dtype=torch.float32, device=model_device)
    POS_W = torch.tensor(float(pos_weight), dtype=torch.float32, device=model_device)

    total_val_time = 0.0
    for ep in range(1, max_epochs + 1):
        model.train()
        total_loss = 0.0
        total_seen = 0
        start_time = time.perf_counter()

        opt.zero_grad(set_to_none=True)
        for step, (xb, yb) in enumerate(train_loader, start=1):
            xb = xb.to(model_device, non_blocking=True)
            yb = yb.to(model_device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=amp):
                logits = model(xb)
                loss_raw = bce(logits, yb)
                w = torch.where(yb > 0.5, POS_W, NEG_W)
                loss = (loss_raw * w).mean() / max(1, grad_accum_steps)

            if amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if step % grad_accum_steps == 0:
                if amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                opt.zero_grad(set_to_none=True)

            total_loss += float(loss.item()) * int(xb.shape[0])
            total_seen += int(xb.shape[0])

        train_time = max(1e-6, time.perf_counter() - start_time)
        samples_sec = total_seen / train_time if total_seen else 0.0
        total_loss /= max(1, total_seen)

        model.eval()
        val_start = time.perf_counter()
        val_probs_list: list[torch.Tensor] = []
        val_y_list: list[torch.Tensor] = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(model_device, non_blocking=True)
                logits = model(xb)
                probs = torch.sigmoid(logits).detach().cpu()
                val_probs_list.append(probs)
                val_y_list.append(yb.detach().cpu())
        total_val_time += time.perf_counter() - val_start

        val_probs = torch.cat(val_probs_list, dim=0) if val_probs_list else torch.zeros((0, 1))
        val_y = torch.cat(val_y_list, dim=0) if val_y_list else torch.zeros((0, 1))
        thr, m, score = find_best_threshold(val_probs, val_y, fpr_penalty=float(fpr_penalty))

        scheduler.step(score)
        cur_lr = opt.param_groups[0]["lr"]

        if ep == 1 or ep % 2 == 0:
            print(
                f"[EP {ep:04d}] loss={total_loss:.4f} | "
                f"BEST(thr={thr:.2f}) acc={m.acc:.3f} prec={m.precision:.3f} rec={m.recall:.3f} "
                f"f1={m.f1:.3f} fpr={m.fpr:.3f} | score={score:.3f} | lr={cur_lr:.6f} "
                f"| {samples_sec:.1f} samples/s"
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
    return best_state, best_thr, best_m, float(best_score), total_val_time


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
    p.add_argument("--build_cache_only", action="store_true", help="построить cache embeddings и выйти")
    p.add_argument("--embed_cache_enabled", type=str, default=None)
    p.add_argument("--embed_cache_dir", type=str, default=None)
    p.add_argument("--cache_rebuild", type=str, default=None)
    p.add_argument("--cache_device", type=str, default=None)
    p.add_argument("--cache_num_workers", type=int, default=None)
    p.add_argument("--cache_max_gb", type=float, default=None)
    p.add_argument("--cache_prune", type=str, default=None)
    p.add_argument("--mining_mode", type=str, default=None, choices=["full", "subset", "scheduled"])
    p.add_argument("--mining_subset_frac", type=float, default=None)
    p.add_argument("--mining_full_every", type=int, default=None)
    p.add_argument("--hard_k_pos", type=int, default=None)
    p.add_argument("--hard_k_neg", type=int, default=None)
    p.add_argument("--hard_repeat_pos", type=int, default=None)
    p.add_argument("--hard_repeat_neg", type=int, default=None)
    p.add_argument("--eval_each_round", type=str, default=None)
    p.add_argument("--full_eval_each_round", type=str, default=None)
    p.add_argument("--full_eval_every", type=int, default=None)
    p.add_argument("--threshold_sweep_enabled", type=str, default=None)
    p.add_argument("--threshold_sweep_every", type=int, default=None)
    p.add_argument("--amp", type=str, default=None)
    p.add_argument("--cudnn_benchmark", type=str, default=None)
    p.add_argument("--pin_memory", type=str, default=None)
    p.add_argument("--prefetch_factor", type=int, default=None)
    p.add_argument("--persistent_workers", type=str, default=None)
    p.add_argument("--num_workers", type=str, default=None)
    p.add_argument("--grad_accum_steps", type=int, default=None)
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
    tr_cfg = _load_yaml(DEFAULT_AGENT_CONFIG).get("wake_word_train", {}) if DEFAULT_AGENT_CONFIG else {}

    # режим: current/full -> split/full
    if args.train_regime is not None:
        mode = "split" if args.train_regime == "current" else "full"
    else:
        mode = args.mode or cfg["mode"]

    max_epochs = int(args.epochs if args.epochs is not None else cfg["epochs"])
    batch_raw = args.batch if args.batch is not None else cfg.get("batch", 128)
    batch = _safe_int(batch_raw, 0)
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
    model_input_sec = float(input_shape[0]) * 0.08
    if abs(model_input_sec - total_sec) > 0.05:
        total_sec = model_input_sec
        total_len = int(SR * total_sec)
        if DEFAULT_AGENT_CONFIG is not None:
            cfg_lock = _load_yaml(DEFAULT_AGENT_CONFIG)
            cfg_lock.setdefault("wake_word", {})
            cfg_lock["wake_word"]["total_sec"] = float(total_sec)
            _save_yaml(DEFAULT_AGENT_CONFIG, cfg_lock)
        print(f"[CFG] total_sec locked to {total_sec:.2f} (effective)")
    print(f"[FEATS] input_shape={input_shape} | ncpu={ncpu} | AudioFeatures device={getattr(F, 'device', feats_device)}")

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

    window_mode_train = str(tr_cfg.get("window_mode_train", "random"))
    window_mode_eval = str(tr_cfg.get("window_mode_eval", "center"))

    mining_mode = args.mining_mode or str(tr_cfg.get("mining_mode", "scheduled"))
    mining_subset_frac = _safe_float(args.mining_subset_frac if args.mining_subset_frac is not None else tr_cfg.get("mining_subset_frac", 0.25), 0.25)
    mining_full_every = _safe_int(args.mining_full_every if args.mining_full_every is not None else tr_cfg.get("mining_full_every", 3), 3)
    hard_k_pos = _safe_int(args.hard_k_pos if args.hard_k_pos is not None else tr_cfg.get("hard_k_pos", 200), 200)
    hard_k_neg = _safe_int(args.hard_k_neg if args.hard_k_neg is not None else tr_cfg.get("hard_k_neg", 800), 800)
    hard_repeat_pos = _safe_int(args.hard_repeat_pos if args.hard_repeat_pos is not None else tr_cfg.get("hard_repeat_pos", 3), 3)
    hard_repeat_neg = _safe_int(args.hard_repeat_neg if args.hard_repeat_neg is not None else tr_cfg.get("hard_repeat_neg", 10), 10)

    eval_each_round = _parse_bool(args.eval_each_round, _parse_bool(tr_cfg.get("eval_each_round", True), True))
    full_eval_each_round = _parse_bool(args.full_eval_each_round, _parse_bool(tr_cfg.get("full_eval_each_round", False), False))
    full_eval_every = _safe_int(args.full_eval_every if args.full_eval_every is not None else tr_cfg.get("full_eval_every", 3), 3)
    threshold_sweep_enabled = _parse_bool(args.threshold_sweep_enabled, _parse_bool(tr_cfg.get("threshold_sweep_enabled", False), False))
    threshold_sweep_every = _safe_int(args.threshold_sweep_every if args.threshold_sweep_every is not None else tr_cfg.get("threshold_sweep_every", 3), 3)

    embed_cache_enabled = _parse_bool(args.embed_cache_enabled, _parse_bool(tr_cfg.get("embed_cache_enabled", True), True))
    embed_cache_dir = Path(args.embed_cache_dir) if args.embed_cache_dir else Path(tr_cfg.get("embed_cache_dir", "cache/embeddings"))
    cache_rebuild = _parse_bool(args.cache_rebuild, _parse_bool(tr_cfg.get("cache_rebuild", False), False))
    cache_device = str(args.cache_device or tr_cfg.get("cache_device", "cpu")).lower()
    if cache_device == "cuda" and not torch.cuda.is_available():
        print("[WARN] cache_device=cuda, но CUDA недоступна -> cache_device=cpu")
        cache_device = "cpu"
    cache_max_gb = _safe_float(args.cache_max_gb if args.cache_max_gb is not None else tr_cfg.get("cache_max_gb", 30), 30.0)
    cache_prune = _parse_bool(args.cache_prune, _parse_bool(tr_cfg.get("cache_prune", True), True))
    if not embed_cache_enabled:
        print("[WARN] embed_cache_enabled=false: принудительно включаю cache для ускорения.")
        embed_cache_enabled = True

    # devices
    if model_device_str == "cuda" and not torch.cuda.is_available():
        print("[WARN] model_device=cuda, но CUDA недоступна -> model_device=cpu")
        model_device_str = "cpu"
    model_device = torch.device(model_device_str)

    def _auto_workers() -> int:
        cpu = os.cpu_count() or 4
        base = max(2, cpu - 2)
        if sys.platform.startswith("win"):
            return min(6, base)
        return min(8, base)

    num_workers_raw = args.num_workers if args.num_workers is not None else tr_cfg.get("num_workers", "auto")
    num_workers = _auto_workers() if str(num_workers_raw).lower() == "auto" else _safe_int(num_workers_raw, _auto_workers())
    cache_num_workers_raw = args.cache_num_workers if args.cache_num_workers is not None else tr_cfg.get("cache_num_workers", num_workers)
    cache_num_workers = _safe_int(cache_num_workers_raw, num_workers)

    is_cuda = model_device.type == "cuda"
    amp = _parse_bool(args.amp, _parse_bool(tr_cfg.get("amp", True), True)) and is_cuda
    cudnn_benchmark = _parse_bool(args.cudnn_benchmark, _parse_bool(tr_cfg.get("cudnn_benchmark", True), True)) and is_cuda
    if is_cuda:
        torch.backends.cudnn.benchmark = cudnn_benchmark

    pin_memory = _parse_bool(args.pin_memory, _parse_bool(tr_cfg.get("pin_memory", True), True)) and is_cuda
    prefetch_factor = _safe_int(args.prefetch_factor if args.prefetch_factor is not None else tr_cfg.get("prefetch_factor", 2), 2)
    persistent_workers = _parse_bool(args.persistent_workers, _parse_bool(tr_cfg.get("persistent_workers", True), True)) and num_workers > 0
    grad_accum_steps = _safe_int(args.grad_accum_steps if args.grad_accum_steps is not None else tr_cfg.get("grad_accum_steps", 1), 1)
    loader_kwargs = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0:
        loader_kwargs["prefetch_factor"] = prefetch_factor
        loader_kwargs["persistent_workers"] = persistent_workers

    if batch <= 0:
        if is_cuda:
            candidates = [1024, 768, 512, 384, 256, 128]
            for cand in candidates:
                try:
                    dummy = torch.zeros((cand, input_shape[0], input_shape[1]), device=model_device)
                    model_probe = DNNWakeword(input_shape=input_shape, layer_size=layer).to(model_device)
                    with torch.amp.autocast("cuda", enabled=amp):
                        _ = model_probe(dummy)
                    batch = cand
                    break
                except RuntimeError:
                    torch.cuda.empty_cache()
                    continue
            if batch <= 0:
                batch = 128
        else:
            batch = 256

    print(f"[TRAIN] device={model_device} mode={mode} epochs={max_epochs} batch={batch} lr={lr} wd={wd} patience={patience}")
    print(f"       neg_weight={neg_weight:.2f} pos_weight={pos_weight:.2f} | fpr_penalty={fpr_penalty:.2f}")
    print(f"       rounds={rounds} | mine_thr={mine_thr:.2f} | mining_mode={mining_mode}")
    print(f"       num_workers={num_workers} pin_memory={pin_memory} amp={amp} grad_accum={grad_accum_steps}")
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

    feature_id = f"oww_{input_shape[0]}x{input_shape[1]}_sr{SR}"
    train_items_base: List[CacheItem] = [
        CacheItem(path=p, label=1, sr=SR, total_sec=total_sec, window_mode=window_mode_train, split="train")
        for p in train_pos
    ] + [
        CacheItem(path=p, label=0, sr=SR, total_sec=total_sec, window_mode=window_mode_train, split="train")
        for p in train_neg_base
    ]
    val_items: List[CacheItem] = [
        CacheItem(path=p, label=1, sr=SR, total_sec=total_sec, window_mode=window_mode_eval, split="val")
        for p in val_pos
    ] + [
        CacheItem(path=p, label=0, sr=SR, total_sec=total_sec, window_mode=window_mode_eval, split="val")
        for p in val_neg
    ]

    if embed_cache_enabled:
        if cache_rebuild and embed_cache_dir.exists():
            for p in embed_cache_dir.rglob("*.pt"):
                p.unlink(missing_ok=True)
        index_start = time.perf_counter()
        hits, misses, cache_compute = build_embedding_cache(
            train_items_base + val_items,
            embed_cache_dir,
            cache_device=cache_device,
            num_workers=cache_num_workers,
            feature_id=feature_id,
            total_sec=total_sec,
            sr=SR,
        )
        index_time = time.perf_counter() - index_start
        if cache_prune:
            prune_cache(embed_cache_dir, cache_max_gb)
        print(f"[CACHE] hits={hits} miss={misses} compute_time={cache_compute:.1f}s index_time={index_time:.1f}s")
        if args.build_cache_only:
            print("[CACHE] build_cache_only -> exit")
            return 0

    val_dataset = EmbeddingDataset(val_items, embed_cache_dir, train=False, feature_id=feature_id)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch,
        shuffle=False,
        **loader_kwargs,
    )

    # hard-mining pools
    hard_pos_idx_pool: List[int] = []
    hard_neg_idx_pool: List[int] = []
    hard_dirs: List[Path] = []

    for r in range(1, rounds + 1):
        print(f"\n========== ROUND {r}/{rounds} ==========")

        index_start = time.perf_counter()
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

        train_items = train_items_base + [
            CacheItem(path=p, label=0, sr=SR, total_sec=total_sec, window_mode=window_mode_train, split="train")
            for p in neg_extra
        ]

        cache_hits = 0
        cache_misses = 0
        cache_compute = 0.0
        if embed_cache_enabled:
            cache_hits, cache_misses, cache_compute = build_embedding_cache(
                train_items,
                embed_cache_dir,
                cache_device=cache_device,
                num_workers=cache_num_workers,
                feature_id=feature_id,
                total_sec=total_sec,
                sr=SR,
            )
        index_time = time.perf_counter() - index_start

        base_indices = list(range(len(train_items)))
        train_dataset = EmbeddingDataset(train_items, embed_cache_dir, train=True, feature_id=feature_id)
        train_indices = base_indices.copy()

        # TRAIN set: augmented base + hard oversample
        if hard_pos_idx_pool or hard_neg_idx_pool:
            hard_pos_idx_pool = hard_pos_idx_pool[: max_copy_pos]
            hard_neg_idx_pool = hard_neg_idx_pool[: max_copy_neg]
            train_indices += hard_pos_idx_pool * hard_repeat_pos
            train_indices += hard_neg_idx_pool * hard_repeat_neg
            print(f"[HARD] добавлено: hard_pos={len(hard_pos_idx_pool)}x{hard_repeat_pos}, hard_neg={len(hard_neg_idx_pool)}x{hard_repeat_neg}")

        max_train = int(len(base_indices) * 4)
        if len(train_indices) > max_train:
            train_indices = random.sample(train_indices, k=max_train)

        train_loader = DataLoader(
            torch.utils.data.Subset(train_dataset, train_indices),
            batch_size=batch,
            shuffle=True,
            **loader_kwargs,
        )

        train_start = time.perf_counter()
        best_state, best_thr, best_m, best_score, val_time = train_one_round_loader(
            train_loader=train_loader,
            val_loader=val_loader,
            input_shape=input_shape,
            layer=layer,
            lr=lr,
            wd=wd,
            max_epochs=max_epochs,
            patience=patience,
            model_device=model_device,
            neg_weight=neg_weight,
            pos_weight=pos_weight,
            fpr_penalty=fpr_penalty,
            no_early_stop=args.no_early_stop,
            amp=amp,
            grad_accum_steps=grad_accum_steps,
        )
        train_time = time.perf_counter() - train_start

        if best_score > best_score_overall:
            best_score_overall = best_score
            best_state_overall = best_state
            best_thr_overall = best_thr
            best_metrics_overall = best_m

        mining_start = time.perf_counter()
        model_cpu = DNNWakeword(input_shape=input_shape, layer_size=layer)
        model_cpu.load_state_dict(best_state)
        model_cpu.eval()
        mining_indices = list(range(len(train_items)))
        if mining_mode in {"subset", "scheduled"}:
            do_full = mining_mode == "full" or (mining_mode == "scheduled" and r % mining_full_every == 0)
            if not do_full:
                subset_n = max(1, int(len(mining_indices) * mining_subset_frac))
                mining_indices = random.sample(mining_indices, k=subset_n)
        if hard_pos_idx_pool or hard_neg_idx_pool:
            mining_indices = list(set(mining_indices + hard_pos_idx_pool + hard_neg_idx_pool))

        all_wavs = [item.path for item in train_items]
        mining_dataset = EmbeddingDataset(train_items, embed_cache_dir, train=False, feature_id=feature_id)
        mining_loader = DataLoader(
            torch.utils.data.Subset(mining_dataset, mining_indices),
            batch_size=batch,
            shuffle=False,
            **loader_kwargs,
        )

        probs_all_list: list[torch.Tensor] = []
        y_all_list: list[torch.Tensor] = []
        with torch.no_grad():
            for xb, yb in mining_loader:
                logits = model_cpu(xb)
                probs_all_list.append(torch.sigmoid(logits).detach().cpu())
                y_all_list.append(yb.detach().cpu())

        probs_all = torch.cat(probs_all_list, dim=0) if probs_all_list else torch.zeros((0, 1))
        y_all = torch.cat(y_all_list, dim=0) if y_all_list else torch.zeros((0, 1))

        hard = select_hard_by_scores(
            probs_all,
            y_all,
            k_pos=hard_k_pos,
            k_neg=hard_k_neg,
        )
        mining_time = time.perf_counter() - mining_start

        print(f"[HARD FOUND] hard_negative(FP)={hard.hard_neg_count} | hard_positive(FN)={hard.hard_pos_count}")
        print(f"[HARD POOL] train_size={len(train_items)} | hard_pos={hard.hard_pos_count} hard_neg={hard.hard_neg_count}")

        # сортировки: NEG по убыванию score (самые опасные), POS по возрастанию (самые провальные)
        if hard.hard_neg_idx:
            hard.hard_neg_idx = sorted(hard.hard_neg_idx, key=lambda i: float(probs_all[i].item()), reverse=True)
        if hard.hard_pos_idx:
            hard.hard_pos_idx = sorted(hard.hard_pos_idx, key=lambda i: float(probs_all[i].item()))

        hard_neg_idx_pool = [mining_indices[i] for i in hard.hard_neg_idx[: max_copy_neg]]
        hard_pos_idx_pool = [mining_indices[i] for i in hard.hard_pos_idx[: max_copy_pos]]

        hard_paths = [train_items[i].path for i in hard_neg_idx_pool + hard_pos_idx_pool]
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

        total_time = index_time + cache_compute + train_time + val_time + mining_time
        print(
            f"TIMING: index={index_time:.1f}s cache_compute={cache_compute:.1f}s "
            f"train={train_time:.1f}s val={val_time:.1f}s mining={mining_time:.1f}s total={total_time:.1f}s"
        )

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
