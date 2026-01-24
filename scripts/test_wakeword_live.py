from __future__ import annotations

import time
from pathlib import Path
import numpy as np
import sounddevice as sd
from openwakeword.model import Model


def main() -> None:
    repo = Path(__file__).resolve().parents[1]
    model_path = repo / "models" / "agent.onnx"  # <-- поменяй если надо

    sample_rate = 16000
    chunk_ms = 20
    chunk_samples = int(sample_rate * chunk_ms / 1000)

    # логика как у тебя в config.yaml
    threshold = 0.35
    patience_frames = 2
    cooldown_ms = 1200
    min_rms = 0.008

    m = Model(wakeword_models=[str(model_path)], inference_framework="onnx", vad_threshold=0.0)
    key = list(m.models.keys())[0]

    window_s = 1.0
    window_samples = int(sample_rate * window_s)
    buffer = np.zeros((0,), dtype=np.int16)

    hits = 0
    cooldown_until = 0.0

    print(f"[LIVE] model={model_path.name} key='{key}' thr={threshold} patience={patience_frames}")
    print("[LIVE] Говори: 'Бивис'")

    with sd.InputStream(samplerate=sample_rate, channels=1, dtype="float32", blocksize=chunk_samples) as stream:
        while True:
            data, _ = stream.read(chunk_samples)
            x = data[:, 0]
            x16 = (np.clip(x, -1, 1) * 32767).astype(np.int16)

            # rms gate
            xf = x16.astype(np.float32) / 32768.0
            rms = float(np.sqrt(np.mean(xf * xf) + 1e-12))
            if rms < min_rms:
                continue

            buffer = np.concatenate([buffer, x16])
            if buffer.size > window_samples:
                buffer = buffer[-window_samples:]

            now = time.time()
            if now < cooldown_until:
                continue

            if buffer.size < window_samples:
                continue

            scores = m.predict(buffer)
            sc = float(scores.get(key, 0.0))

            print(f"\rscore={sc:.3f} rms={rms:.4f}    ", end="")

            if sc >= threshold:
                hits += 1
            else:
                hits = 0

            if hits >= patience_frames:
                hits = 0
                cooldown_until = now + cooldown_ms / 1000.0
                print(f"\n✅ DETECTED! score={sc:.3f}")
                print("[LIVE] снова слушаю...")

if __name__ == "__main__":
    main()
