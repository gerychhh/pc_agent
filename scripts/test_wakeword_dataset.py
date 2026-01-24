from __future__ import annotations

"""scripts/test_wakeword_dataset.py

Единый тест по датасету (data/positive + data/negative),
НО: все параметры (модель, threshold, total_sec, patience_frames и т.д.)
берутся ИСКЛЮЧИТЕЛЬНО из voice_agent/config.yaml.

Зачем:
- чтобы у тебя не было 5 разных скриптов с 5 разными порогами
- чтобы результат теста соответствовал тому, как агент работает в LIVE

Запуск:
    python scripts/test_wakeword_dataset.py
"""

from pathlib import Path
import wave
import math

import numpy as np

try:
    import yaml
except Exception:
    yaml = None

try:
    from openwakeword.model import Model
except Exception:
    Model = None


SR = 16000
CHUNK_SIZE = 1280  # 80ms @ 16kHz (рекомендованный streaming chunk)


def _load_yaml(path: Path) -> dict:
    if yaml is None or not path.exists():
        return {}
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def _resolve_config_path(repo: Path) -> Path:
    p1 = repo / "voice_agent" / "config.yaml"
    if p1.exists():
        return p1
    p2 = repo / "config.yaml"
    if p2.exists():
        return p2
    raise FileNotFoundError("config.yaml not found (expected voice_agent/config.yaml)")


def _read_wav_int16(path: Path) -> np.ndarray:
    with wave.open(str(path), "rb") as wf:
        frames = wf.readframes(wf.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16)
    return audio


def _pad_or_trim(audio: np.ndarray, total_len: int) -> np.ndarray:
    if audio.shape[0] >= total_len:
        return audio[:total_len].astype(np.int16, copy=False)
    out = np.zeros((total_len,), dtype=np.int16)
    out[: audio.shape[0]] = audio
    return out


def _rms(audio_int16: np.ndarray) -> float:
    if audio_int16.size == 0:
        return 0.0
    x = audio_int16.astype(np.float32) / 32768.0
    return float(math.sqrt(float(np.mean(x * x)) + 1e-12))


def _score_streaming(
    model: Model,
    audio_int16: np.ndarray,
    model_name: str,
    *,
    threshold: float,
    patience_frames: int,
    min_rms_gate: float,
) -> tuple[bool, float]:
    """Streaming-score как в реальном времени.

    Возвращает:
        detected(bool), max_score(float)
    """
    hits = 0
    max_s = 0.0

    # IMPORTANT: reset состояние модели между файлами
    if hasattr(model, "reset"):
        try:
            model.reset()  # type: ignore[attr-defined]
        except Exception:
            pass

    n = int(audio_int16.shape[0])
    for i in range(0, n, CHUNK_SIZE):
        chunk = audio_int16[i : i + CHUNK_SIZE]
        if chunk.shape[0] < CHUNK_SIZE:
            # добьём нулями, чтобы размер был стабильный
            chunk = _pad_or_trim(chunk, CHUNK_SIZE)

        # RMS gate как в агенте (если тишина — не учитываем)
        if _rms(chunk) < min_rms_gate:
            hits = 0
            continue

        scores = model.predict(chunk)
        s = float(scores.get(model_name, 0.0))
        max_s = max(max_s, s)

        if s >= threshold:
            hits += 1
        else:
            hits = 0

        if hits >= max(1, int(patience_frames)):
            return True, max_s

    return False, max_s


def main() -> None:
    if Model is None:
        print("[ERR] openwakeword не импортируется. Установи: pip install openwakeword")
        return

    repo = Path(__file__).resolve().parents[1]
    cfg_path = _resolve_config_path(repo)
    cfg = _load_yaml(cfg_path)

    ww = cfg.get("wake_word", {}) or {}

    # модель
    model_paths = ww.get("model_paths", [])
    if isinstance(model_paths, list) and model_paths:
        model_path = Path(str(model_paths[0]))
        if not model_path.is_absolute():
            model_path = cfg_path.parent / model_path
    else:
        model_path = repo / "models" / "agent.onnx"

    if not model_path.exists():
        print(f"[ERR] Модель не найдена: {model_path}")
        return

    threshold = float(ww.get("threshold", 0.7))
    patience_frames = int(ww.get("patience_frames", 3))
    total_sec = float(ww.get("total_sec", 2.0))
    min_rms_gate = float(ww.get("min_rms", 0.0025))

    total_len = int(SR * total_sec)

    pos_dir = repo / "data" / "positive"
    neg_dir = repo / "data" / "negative"

    pos_files = sorted(pos_dir.glob("*.wav"))
    neg_files = sorted(neg_dir.glob("*.wav"))

    if not pos_files or not neg_files:
        print("[ERR] Не нашёл wav в data/positive или data/negative")
        return

    model = Model(wakeword_models=[str(model_path)], inference_framework=ww.get("inference_framework", "onnx"), vad_threshold=0.0)
    model_name = str(ww.get("model_name", ""))
    if not model_name:
        # берём первый ключ, который есть внутри модели
        model_name = list(model.models.keys())[0]

    print(f"\n[CFG] {cfg_path}")
    print(f"[MODEL] {model_path.name} | key='{model_name}'")
    print(f"[PARAMS] total_sec={total_sec:.2f} | threshold={threshold:.3f} | patience_frames={patience_frames} | min_rms={min_rms_gate:.4f}")
    print(f"[DATA] POS={len(pos_files)} NEG={len(neg_files)}")

    # positive
    tp = 0
    fn = 0
    worst_pos: list[tuple[float, str]] = []
    for f in pos_files:
        a = _pad_or_trim(_read_wav_int16(f), total_len)
        ok, max_s = _score_streaming(
            model,
            a,
            model_name,
            threshold=threshold,
            patience_frames=patience_frames,
            min_rms_gate=min_rms_gate,
        )
        if ok:
            tp += 1
        else:
            fn += 1
            worst_pos.append((max_s, f.name))

    # negative
    tn = 0
    fp = 0
    worst_neg: list[tuple[float, str]] = []
    for f in neg_files:
        a = _pad_or_trim(_read_wav_int16(f), total_len)
        ok, max_s = _score_streaming(
            model,
            a,
            model_name,
            threshold=threshold,
            patience_frames=patience_frames,
            min_rms_gate=min_rms_gate,
        )
        if not ok:
            tn += 1
        else:
            fp += 1
            worst_neg.append((max_s, f.name))

    total = tp + fn + tn + fp
    acc = (tp + tn) / max(total, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    f1 = (2 * precision * recall) / max(precision + recall, 1e-12)

    print("\n[RESULT]")
    print(f"✅ POS поймал: {tp}/{tp + fn} | ❌ пропустил: {fn}")
    print(f"✅ NEG игнор:  {tn}/{tn + fp} | ❌ ложняки:  {fp}")
    print(f"acc={acc:.3f} | precision={precision:.3f} | recall={recall:.3f} | f1={f1:.3f} | fpr={fpr:.3f}")

    worst_pos.sort(key=lambda x: x[0])
    worst_neg.sort(key=lambda x: -x[0])

    print("\n[WORST POS] (максимальный score был низкий)")
    for sc, name in worst_pos[:10]:
        print(f"  {sc:.3f}  {name}")

    print("\n[WORST NEG] (максимальный score был высокий)")
    for sc, name in worst_neg[:10]:
        print(f"  {sc:.3f}  {name}")


if __name__ == "__main__":
    main()
