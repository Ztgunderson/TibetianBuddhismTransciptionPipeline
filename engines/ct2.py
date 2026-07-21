"""Engine B: faster-whisper / CTranslate2 (INT8).

The TensorRT-adjacent second arm. Built and run only after Loop 1's ggml pass is complete
and on schedule (see the approved plan's nested fallbacks). Its distinctive Loop 2 role is
INT8 quantization via a CALIBRATION CACHE — the native NVIDIA-stack expression of
language-aware quantization: build one cache from English audio and one from Tibetan audio,
compare at identical bits.

`faster_whisper` is imported lazily inside __init__ so that Engine-A-only runs never require
it to be installed.
"""

from __future__ import annotations

import time
import wave

from engines.base import Engine, Transcription


def _wav_seconds(path: str) -> float:
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


class Ct2Engine(Engine):
    name = "ct2"

    def __init__(
        self,
        model_path: str,
        compute_type: str = "int8",   # 'int8' | 'float16' — the quant axis for this engine
        device: str = "cuda",
        cpu_threads: int = 0,
    ):
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:  # pragma: no cover - Engine B not yet built
            raise ImportError(
                "faster-whisper not installed. Engine B (CTranslate2/INT8) is built only "
                "after Loop 1; run models/build_ct2.sh inside the jetson-container first."
            ) from exc
        self.model_path = model_path
        self.compute_type = compute_type
        self._model = WhisperModel(
            model_path, device=device, compute_type=compute_type, cpu_threads=cpu_threads
        )

    def transcribe(self, audio_path: str, lang: str) -> Transcription:
        audio_seconds = _wav_seconds(audio_path)
        start = time.perf_counter()
        segments, _ = self._model.transcribe(audio_path, language=lang, beam_size=5)
        text = " ".join(seg.text.strip() for seg in segments).strip()
        compute_seconds = time.perf_counter() - start
        return Transcription(text=text, audio_seconds=audio_seconds, compute_seconds=compute_seconds)
