from __future__ import annotations

import shutil
from pathlib import Path
from typing import Iterable


def _pick_source(text: str | Iterable[str]) -> Path:
    if isinstance(text, str):
        texts = [text]
    else:
        texts = list(text)
    lowered = [item.lower() for item in texts]
    if any("агент" in item for item in lowered):
        return Path("data/positive")
    return Path("data/negative")


def generate_samples(
    text: str | Iterable[str],
    max_samples: int,
    batch_size: int | None = None,
    noise_scales: Iterable[float] | None = None,
    noise_scale_ws: Iterable[float] | None = None,
    length_scales: Iterable[float] | None = None,
    output_dir: str | Path = ".",
    auto_reduce_batch_size: bool = False,
    file_names: Iterable[str] | None = None,
) -> None:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    source_dir = _pick_source(text)
    if not source_dir.exists():
        raise FileNotFoundError(f"Source directory not found: {source_dir}")

    candidates = sorted(source_dir.glob("*.wav"))
    if not candidates:
        raise FileNotFoundError(f"No .wav files found in {source_dir}")

    if file_names:
        names = list(file_names)
    else:
        names = [f"sample_{idx + 1:05d}.wav" for idx in range(max_samples)]

    for idx in range(min(max_samples, len(names))):
        src = candidates[idx % len(candidates)]
        dst = output_path / names[idx]
        shutil.copy2(src, dst)
