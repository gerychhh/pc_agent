from __future__ import annotations

import time
from dataclasses import dataclass
import logging
import os
import ctypes
from pathlib import Path

import numpy as np
from packaging import version
# NOTE: Import faster_whisper lazily after setting DLL search paths on Windows.
# from faster_whisper import WhisperModel

from .bus import Event, EventBus


@dataclass(frozen=True)
class AsrConfig:
    model: str
    device: str
    compute_type: str
    beam_size: int
    language: str
    max_utterance_s: int
    partial_interval_ms: int
    partial_min_delta: int
    min_partial_s: float
    sample_rate: int
    no_speech_threshold: float
    log_prob_threshold: float
    compression_ratio_threshold: float
    min_buffer_s: float


class FasterWhisperASR:
    def __init__(self, config: AsrConfig, bus: EventBus) -> None:
        self.config = config
        self.bus = bus
        self.logger = logging.getLogger("voice_agent")
        self._fallback_attempted = False
        self._prepare_cuda_env()

        device = str(config.device or "auto").lower()
        compute_type = str(config.compute_type or "float16").lower()

        if device == "auto":
            # Prefer CUDA if available; otherwise CPU.
            try:
                import torch  # type: ignore
                device = "cuda" if torch.cuda.is_available() else "cpu"
            except Exception:
                device = "cpu"

        if device == "cuda":
            ok, reason = self._cuda_preflight_ok()
            if not ok:
                self.logger.warning(
                    "ASR CUDA preflight failed (%s). Switching ASR to CPU to avoid native crash. "
                    "Tip: ctranslate2>=4.5 typically requires cuDNN v9 and CUDA>=12.3 on Windows.",
                    reason,
                )
                device = "cpu"
                compute_type = "int8"

        self._load_model(device, compute_type)
        self._buffer: list[np.ndarray] = []
        self._last_partial: str = ""
        self._last_partial_emit = 0.0
        self._active = False
        self._speech_start_ts: float | None = None
        self._chunk_log_counter = 0

    def _load_model(self, device: str, compute_type: str) -> None:
        try:
            from faster_whisper import WhisperModel  # lazy import
            self.model = WhisperModel(
                self.config.model,
                device=device,
                compute_type=compute_type,
            )
            self.logger.info("ASR model loaded on %s (%s).", device, compute_type)
            self.logger.info("ASR model=%s language=%s beam=%s", self.config.model, self.config.language, self.config.beam_size)
        except (RuntimeError, OSError) as exc:
            message = str(exc).lower()
            if device == "cuda" and ("cublas" in message or "cuda" in message or "cudnn" in message or "symbol" in message or "error code 127" in message):
                self.logger.warning(
                    "ASR GPU init failed (%s). Falling back to CPU.",
                    exc,
                )
                self._fallback_attempted = True
                self.model = WhisperModel(
                    self.config.model,
                    device="cpu",
                    compute_type="int8",
                )
                self.logger.info("ASR model loaded on cpu (int8).")
            else:
                raise

    def _prepare_cuda_env(self) -> None:
        if os.name != "nt":
            return

        # Ensure PyTorch-bundled CUDA/cuDNN DLLs are discoverable (torch\lib contains cudnn_*_64_8.dll).
        try:
            import torch  # type: ignore
            torch_lib = Path(torch.__file__).resolve().parent / "lib"
            if torch_lib.is_dir():
                p = str(torch_lib)
                if p not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = f"{p};{os.environ.get('PATH','')}"
                try:
                    os.add_dll_directory(p)
                except OSError:
                    pass
        except Exception:
            pass

        # Ensure ctranslate2 package directory is discoverable (some wheels load DLLs relative to it).
        try:
            import ctranslate2  # type: ignore
            ct2_dir = Path(ctranslate2.__file__).resolve().parent
            if ct2_dir.is_dir():
                p = str(ct2_dir)
                if p not in os.environ.get("PATH", ""):
                    os.environ["PATH"] = f"{p};{os.environ.get('PATH','')}"
                try:
                    os.add_dll_directory(p)
                except OSError:
                    pass
        except Exception:
            pass
        cuda_path = os.environ.get("CUDA_PATH")
        candidates = []
        if cuda_path:
            candidates.append(Path(cuda_path) / "bin")
        toolkit_root = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA")
        if toolkit_root.exists():
            for entry in sorted(toolkit_root.glob("v*")):
                candidates.append(entry / "bin")
        for candidate in candidates:
            if not candidate.exists():
                continue
            path_str = str(candidate)
            if path_str not in os.environ.get("PATH", ""):
                os.environ["PATH"] = f"{path_str};{os.environ.get('PATH', '')}"
            try:
                os.add_dll_directory(path_str)
            except OSError:
                continue

    def _find_dll_in_paths(self, dll_names: list[str]) -> str | None:
        paths = [p for p in os.environ.get("PATH", "").split(";") if p]

        # Common NVIDIA cuDNN location on Windows:
        if os.name == "nt":
            cudnn_root = Path("C:/Program Files/NVIDIA/CUDNN")
            if cudnn_root.exists():
                # Prefer newest v9.* folders and newest CUDA subfolder inside bin.
                for vdir in sorted(cudnn_root.glob("v9*"), reverse=True):
                    bin_dir = vdir / "bin"
                    if bin_dir.is_dir():
                        # Sometimes packaged as bin/<cuda-version>/
                        for sub in sorted(bin_dir.glob("*"), reverse=True):
                            if sub.is_dir():
                                paths.insert(0, str(sub))
                        paths.insert(0, str(bin_dir))

        for p in paths:
            base = Path(p)
            for name in dll_names:
                cand = base / name
                if cand.exists():
                    return str(cand)
        return None

    def _dll_has_symbol(self, dll_path: str, symbol: str) -> bool:
        try:
            lib = ctypes.WinDLL(dll_path)
            getattr(lib, symbol)
            return True
        except OSError:
            return False
        except AttributeError:
            return False

    def _cuda_preflight_ok(self) -> tuple[bool, str]:
        # Avoid native crash on Windows when CUDA/cuDNN/ctranslate2 are mismatched.
        if os.name != "nt":
            return True, ""

        try:
            import ctranslate2  # type: ignore
            ct2_ver = version.parse(getattr(ctranslate2, "__version__", "0"))
        except Exception as e:
            return False, f"ctranslate2 import failed: {e}"

        torch_cuda = ""
        try:
            import torch  # type: ignore
            torch_cuda = str(getattr(torch.version, "cuda", "") or "")
        except Exception:
            pass

        # Heuristic derived from known faster-whisper/ctranslate2 compatibility notes:
        # CT2>=4.5.0 uses cuDNN v9 and typically requires CUDA>=12.3.
        if ct2_ver >= version.parse("4.5.0"):
            dll = self._find_dll_in_paths([
                "cudnn64_9.dll",
                "cudnn_ops64_9.dll",
                "cudnn_ops_infer64_9.dll",
                "cudnn_cnn_infer64_9.dll",
            ])
            if not dll:
                return False, "cuDNN v9 DLL not found on PATH (e.g., cudnn64_9.dll / cudnn_ops64_9.dll)"
            sym_dll = self._find_dll_in_paths(["cudnn64_9.dll"]) or dll
            if not self._dll_has_symbol(sym_dll, "cudnnGetLibConfig"):
                return False, f"cuDNN DLL missing symbol cudnnGetLibConfig ({Path(sym_dll).name})"
            try:
                if torch_cuda and version.parse(torch_cuda) < version.parse("12.3"):
                    return False, f"torch CUDA {torch_cuda} < 12.3 while ctranslate2 {ct2_ver} expects >=12.3"
            except Exception:
                pass

        return True, ""


    def reset(self) -> None:
        self._buffer = []
        self._last_partial = ""
        self._last_partial_emit = 0.0
        self._active = False
        self._speech_start_ts = None
        self._chunk_log_counter = 0

    def speech_start(self) -> None:
        self.reset()
        self._active = True
        self._speech_start_ts = time.monotonic()

    def speech_end(self, ts: float) -> None:
        if not self._active:
            return
        self._finalize(ts)

    def accept_audio(self, chunk: np.ndarray, ts: float) -> None:
        if not self._active:
            return
        # Normalize chunk to mono 1-D int16.
        # AudioCapture publishes (frames, channels) int16; preroll may be (frames,) int16.
        arr = np.asarray(chunk)
        if arr.size == 0:
            return
        if arr.ndim > 1:
            # Take first channel
            arr = arr[:, 0]
        # Ensure int16
        if arr.dtype != np.int16:
            # If float audio is in [-1, 1], scale; otherwise cast.
            try:
                max_abs = float(np.max(np.abs(arr)))
            except Exception:
                max_abs = 0.0
            if max_abs <= 1.5:
                arr = (arr.astype(np.float32) * 32768.0).clip(-32768, 32767).astype(np.int16)
            else:
                arr = arr.astype(np.int16, copy=False)

        self._buffer.append(arr.copy())
        if self.logger.isEnabledFor(logging.DEBUG):
            self._chunk_log_counter += 1
            if self._chunk_log_counter % 25 == 0:
                self.logger.debug("ASR buffer chunks=%d", len(self._buffer))
        if self._speech_start_ts and (ts - self._speech_start_ts) >= self.config.max_utterance_s:
            self.logger.info("ASR max utterance reached, forcing final.")
            self._finalize(ts)
            return
        if self._should_emit_partial(ts):
            if self._buffer_duration_s() < self.config.min_partial_s:
                return
            text = self._transcribe(self._buffer)
            if self._is_significant_partial(text):
                self._last_partial = text
                self._last_partial_emit = ts
                self.bus.publish(Event("asr.partial", {"text": text, "ts": ts, "stability": 0.5}))

    def _finalize(self, ts: float) -> None:
        if self._buffer_duration_s() < self.config.min_buffer_s:
            self.logger.info("ASR buffer too short (%.2fs), dropping.", self._buffer_duration_s())
            self.reset()
            return
        text = self._transcribe(self._buffer)
        if self._last_partial and len(self._last_partial) > len(text) + 2:
            self.logger.info("ASR final shorter than last partial, using partial.")
            text = self._last_partial
        if text:
            self.bus.publish(Event("asr.final", {"text": text, "ts": ts}))
        self.reset()

    def _should_emit_partial(self, ts: float) -> bool:
        return (ts - self._last_partial_emit) * 1000.0 >= self.config.partial_interval_ms

    def _is_significant_partial(self, text: str) -> bool:
        if not text:
            return False
        if text == self._last_partial:
            return False
        return abs(len(text) - len(self._last_partial)) >= self.config.partial_min_delta

    def _buffer_duration_s(self) -> float:
        if not self._buffer:
            return 0.0
        total_samples = 0
        for chunk in self._buffer:
            total_samples += int(chunk.shape[0])
        return total_samples / float(self.config.sample_rate)

    def _run_transcribe(self, audio: np.ndarray) -> str:
        segments, _info = self.model.transcribe(
            audio,
            language=self.config.language,
            beam_size=self.config.beam_size,
            vad_filter=False,
            no_speech_threshold=self.config.no_speech_threshold,
            log_prob_threshold=self.config.log_prob_threshold,
            compression_ratio_threshold=self.config.compression_ratio_threshold,
        )
        return "".join(segment.text for segment in segments).strip()
    def _transcribe(self, chunks: list[np.ndarray]) -> str:
        if not chunks:
            return ""
        audio = np.concatenate(chunks, axis=0).astype(np.float32) / 32768.0
        if audio.ndim > 1:
            audio = audio[:, 0]

        def _needs_cpu_fallback(msg: str) -> bool:
            m = msg.lower()
            return (
                "cudnn_ops_infer64_8.dll" in m
                or "cudnn" in m
                or "cublas" in m
                or "cuda" in m
                or "error code 127" in m
            )

        try:
            return self._run_transcribe(audio)
        except Exception as exc:
            msg = str(exc)
            if not self._fallback_attempted and _needs_cpu_fallback(msg):
                self.logger.warning(
                    "ASR transcribe failed on GPU (%s). Falling back to CPU. "
                    "If you want GPU, ensure torch\\lib is on PATH and cuDNN DLLs like cudnn_ops_infer64_8.dll are available.",
                    msg,
                )
                self._fallback_attempted = True
                try:
                    self._load_model("cpu", "int8")
                    return self._run_transcribe(audio)
                except Exception as retry_exc:
                    self.logger.error("ASR transcribe failed after CPU fallback: %s", retry_exc)
                    return ""
            self.logger.error("ASR transcribe failed: %s", msg)
            return ""
