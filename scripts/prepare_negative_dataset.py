from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import tempfile
import wave
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np

TARGET_SR = 16000
EXTS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".opus")


# =========================
# Utils
# =========================
def has_ffmpeg() -> bool:
    try:
        p = subprocess.run(["ffmpeg", "-version"], capture_output=True, text=True)
        return p.returncode == 0
    except Exception:
        return False


def iter_audio_files(root: Path) -> Iterable[Path]:
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in EXTS:
            yield p


def probe_wav_header(path: Path) -> Tuple[int, int, int, int]:
    """
    Быстрая проверка WAV без чтения всего файла:
    returns: sr, channels, sampwidth, nframes
    """
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
    return sr, ch, sw, n


def read_wav_int16_mono(path: Path) -> Tuple[np.ndarray, int, int, int]:
    """
    Читает WAV PCM16 и возвращает:
    audio(int16 mono), sr, channels(original), sampwidth
    """
    with wave.open(str(path), "rb") as wf:
        sr = wf.getframerate()
        ch = wf.getnchannels()
        sw = wf.getsampwidth()
        n = wf.getnframes()
        raw = wf.readframes(n)

    if sw != 2:
        raise ValueError(f"expected PCM16 (sampwidth=2), got sampwidth={sw}")

    audio = np.frombuffer(raw, dtype=np.int16)

    if ch > 1:
        audio = audio.reshape(-1, ch).mean(axis=1).astype(np.int16)

    return audio, sr, ch, sw


def write_wav_int16_mono(path: Path, audio: np.ndarray, sr: int = TARGET_SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    audio = np.asarray(audio, dtype=np.int16)

    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(audio.tobytes())


def ffmpeg_convert_to_wav16k_mono(src: Path, dst: Path) -> None:
    """
    Любой вход -> WAV mono 16k PCM16
    """
    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(src),
        "-ac",
        "1",
        "-ar",
        str(TARGET_SR),
        "-sample_fmt",
        "s16",
        str(dst),
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        err = (p.stderr or "")[:2500]
        raise RuntimeError(f"ffmpeg failed:\n{err}")


def rms_float01(audio_i16: np.ndarray) -> float:
    x = audio_i16.astype(np.float32) / 32768.0
    return float(np.sqrt(np.mean(x * x) + 1e-12))


def split_into_chunks(audio: np.ndarray, chunk_len: int, hop_len: int, pad_last: bool) -> List[np.ndarray]:
    """
    Нарезка на много кусков:
    chunk_len = длина куска
    hop_len   = шаг
    """
    n = int(len(audio))
    chunks: List[np.ndarray] = []

    if n <= 0:
        return chunks

    if n < chunk_len:
        if pad_last:
            out = np.zeros(chunk_len, dtype=np.int16)
            out[:n] = audio
            chunks.append(out)
        return chunks

    start = 0
    while start + chunk_len <= n:
        chunks.append(audio[start : start + chunk_len])
        start += hop_len

    # хвост
    if pad_last and start < n:
        tail = audio[start:n]
        out = np.zeros(chunk_len, dtype=np.int16)
        out[: len(tail)] = tail
        chunks.append(out)

    return chunks


def find_next_index(out_dir: Path, prefix: str) -> int:
    """
    Чтобы не перетирать уже существующие файлы.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)\.wav$", re.IGNORECASE)
    mx = 0
    for p in out_dir.glob(f"{prefix}_*.wav"):
        m = pat.match(p.name)
        if m:
            mx = max(mx, int(m.group(1)))
    return mx + 1


def wav_is_16k_pcm16(path: Path) -> bool:
    """
    Только формат: wav + sr=16000 + PCM16 (mono/stereo допустим)
    """
    if path.suffix.lower() != ".wav":
        return False
    try:
        sr, ch, sw, n = probe_wav_header(path)
        return (sr == TARGET_SR and sw == 2)
    except Exception:
        return False


def wav_is_already_ready_exact(path: Path, chunk_len: int) -> bool:
    """
    Идеальный клип: wav, pcm16, 16k, mono, длина ровно chunk_len (2.0 сек)
    """
    if path.suffix.lower() != ".wav":
        return False
    try:
        sr, ch, sw, n = probe_wav_header(path)
        if sr != TARGET_SR:
            return False
        if sw != 2:
            return False
        if ch != 1:
            return False
        if n != chunk_len:
            return False
        return True
    except Exception:
        return False


def safe_delete_file(p: Path) -> bool:
    """
    Аккуратное удаление файла.
    Возвращает True если удалили, иначе False.
    """
    try:
        if p.exists() and p.is_file():
            p.unlink()
            return True
    except Exception:
        return False
    return False


# =========================
# Main
# =========================
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=str, required=True, help="папка с исходным NEG датасетом")
    ap.add_argument("--out_dir", type=str, required=True, help="куда сохранить готовые NEG клипы (data/negative)")
    ap.add_argument("--chunk_sec", type=float, default=2.0, help="длина клипа (сек)")
    ap.add_argument("--hop_sec", type=float, default=2.0, help="шаг нарезки (сек) 2.0=без overlap, 1.0=50% overlap")
    ap.add_argument("--min_rms", type=float, default=0.0, help="выкидывать слишком тихие клипы (0 = не фильтровать)")
    ap.add_argument("--pad_last", action="store_true", help="последний хвост дополнять нулями до chunk")
    ap.add_argument("--max_per_file", type=int, default=0, help="лимит клипов с одного файла (0=без лимита)")
    ap.add_argument("--max_total", type=int, default=0, help="общий лимит клипов (0=без лимита)")
    ap.add_argument("--prefix", type=str, default="neg", help="префикс итоговых файлов")

    # поведение для "идеальных" готовых клипов
    ap.add_argument("--copy_ok", action="store_true", help="идеальные 2с 16k mono PCM16 копировать как есть")
    ap.add_argument("--skip_ok", action="store_true", help="идеальные клипы просто пропускать (по умолчанию так)")

    # конвертация
    ap.add_argument("--force_ffmpeg", action="store_true", help="всегда гнать через ffmpeg (даже если wav 16k pcm16)")
    ap.add_argument("--no_ffmpeg", action="store_true", help="запретить ffmpeg (только wav 16k pcm16 будут обработаны)")

    # УДАЛЕНИЕ ПЛОХИХ ИСХОДНИКОВ
    ap.add_argument(
        "--delete_bad",
        action="store_true",
        help="удалять исходные 'плохие' файлы после успешной обработки (не идеальные 2с 16k mono PCM16)",
    )

    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()

    if not in_dir.exists():
        print(f"[ERR] in_dir не существует: {in_dir}")
        return 2

    if args.no_ffmpeg:
        ff_ok = False
    else:
        ff_ok = has_ffmpeg()

    if not ff_ok and not args.no_ffmpeg:
        print("[ERR] ffmpeg не найден. Для 32k/48k и mp3 он обязателен.")
        print("      Проверь: ffmpeg -version")
        return 2

    chunk_len = int(round(args.chunk_sec * TARGET_SR))
    hop_len = int(round(args.hop_sec * TARGET_SR))
    hop_len = max(1, hop_len)

    print(f"[CFG] chunk_sec={args.chunk_sec} hop_sec={args.hop_sec} chunk_len={chunk_len} hop_len={hop_len}")
    print(f"[CFG] min_rms={args.min_rms} pad_last={args.pad_last}")
    print(f"[CFG] copy_ok={args.copy_ok} skip_ok={args.skip_ok} force_ffmpeg={args.force_ffmpeg} no_ffmpeg={args.no_ffmpeg}")
    print(f"[CFG] delete_bad={args.delete_bad}")
    print(f"[IO ] in={in_dir} -> out={out_dir}")

    files = list(iter_audio_files(in_dir))
    if not files:
        print("[ERR] не найдено аудиофайлов")
        return 2

    idx = find_next_index(out_dir, args.prefix)

    seen = 0
    saved = 0

    copied_ok = 0
    skipped_ok = 0

    converted_files = 0
    processed_as_wav_direct = 0

    chunked_files = 0
    skipped_quiet = 0
    errors = 0

    deleted_bad_files = 0
    delete_failed = 0

    for src in files:
        seen += 1
        per_file_saved = 0

        try:
            # это "идеальный" готовый клип?
            is_ok_exact = wav_is_already_ready_exact(src, chunk_len)

            # 0) Если уже ИДЕАЛЬНЫЙ клип (ровно 2с 16k mono PCM16)
            if is_ok_exact and not args.force_ffmpeg:
                if args.copy_ok:
                    out_name = f"{args.prefix}_{idx:08d}.wav"
                    idx += 1
                    shutil.copy2(src, out_dir / out_name)
                    saved += 1
                    copied_ok += 1
                else:
                    skipped_ok += 1

                if seen % 100 == 0:
                    print(
                        f"[PROG] {seen}/{len(files)} saved={saved} conv={converted_files} direct={processed_as_wav_direct} "
                        f"ok_copy={copied_ok} ok_skip={skipped_ok} del_bad={deleted_bad_files} err={errors}"
                    )

                if args.max_total and saved >= args.max_total:
                    break
                continue

            # 1) Определяем источник данных для нарезки:
            #    - если WAV 16k PCM16 (даже если длинный) -> читаем напрямую, НЕ ffmpeg
            #    - иначе -> ffmpeg -> tmp.wav -> читаем
            audio: np.ndarray

            use_direct_wav = (not args.force_ffmpeg) and wav_is_16k_pcm16(src)

            if use_direct_wav:
                audio, sr, ch, sw = read_wav_int16_mono(src)
                if sr != TARGET_SR:
                    raise ValueError(f"wav direct but SR={sr} != {TARGET_SR}")
                processed_as_wav_direct += 1
            else:
                if args.no_ffmpeg:
                    raise ValueError("needs ffmpeg convert but --no_ffmpeg enabled")
                with tempfile.TemporaryDirectory() as td:
                    tmp = Path(td) / "tmp.wav"
                    ffmpeg_convert_to_wav16k_mono(src, tmp)
                    audio, sr, ch, sw = read_wav_int16_mono(tmp)
                converted_files += 1

            # 2) Нарезка на куски
            chunks = split_into_chunks(audio, chunk_len=chunk_len, hop_len=hop_len, pad_last=args.pad_last)
            if len(chunks) > 1:
                chunked_files += 1

            # 3) Сохранение кусков
            for c in chunks:
                if args.min_rms > 0.0:
                    if rms_float01(c) < args.min_rms:
                        skipped_quiet += 1
                        continue

                out_name = f"{args.prefix}_{idx:08d}.wav"
                idx += 1
                write_wav_int16_mono(out_dir / out_name, c, sr=TARGET_SR)
                saved += 1
                per_file_saved += 1

                if args.max_total and saved >= args.max_total:
                    break
                if args.max_per_file and per_file_saved >= args.max_per_file:
                    break

            # 4) Удаление "плохого" исходника после успешного сохранения
            # "плохой" = НЕ идеальный 2с 16k mono PCM16
            if args.delete_bad and (not is_ok_exact) and per_file_saved > 0:
                if safe_delete_file(src):
                    deleted_bad_files += 1
                else:
                    delete_failed += 1

            if seen % 100 == 0:
                print(
                    f"[PROG] {seen}/{len(files)} saved={saved} conv={converted_files} direct={processed_as_wav_direct} "
                    f"ok_copy={copied_ok} ok_skip={skipped_ok} del_bad={deleted_bad_files} err={errors}"
                )

            if args.max_total and saved >= args.max_total:
                break

        except Exception as e:
            errors += 1
            print(f"[WARN] skip '{src.name}': {str(e)[:200]}")

    print("\n[DONE]")
    print(f"files_seen             = {seen}")
    print(f"saved_clips            = {saved}")
    print(f"converted_files(ffmpeg) = {converted_files}  (неправильные -> 16k mono PCM16)")
    print(f"direct_wav_processed   = {processed_as_wav_direct}  (wav 16k pcm16 -> без ffmpeg)")
    print(f"chunked_files          = {chunked_files}  (длинные -> много кусков)")
    print(f"copied_ok_files        = {copied_ok}  (идеальные 2с -> копия)")
    print(f"skipped_ok_files       = {skipped_ok}  (идеальные 2с -> скип)")
    print(f"skipped_quiet_clips    = {skipped_quiet}  (тихо по min_rms)")
    print(f"deleted_bad_files      = {deleted_bad_files}  (исходники удалены после успешной обработки)")
    print(f"delete_failed          = {delete_failed}  (не смог удалить)")
    print(f"errors                 = {errors}")
    print(f"output_dir             = {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
