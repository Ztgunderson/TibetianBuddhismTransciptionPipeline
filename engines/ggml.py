"""Engine A: whisper.cpp (ggml) built with CUDA.

Invokes the `whisper-cli` binary produced by `models/build.sh` as a subprocess, one call per
utterance, parsing plain-text output. Subprocess rather than bindings keeps this identical to
how the model runs in a real edge deployment and avoids pinning a Python-side whisper.cpp
wheel to the Jetson's CUDA build.

Configuration (all from env / constructor, never hardcoded — see docs/design principles):
  bin_path : path to whisper-cli inside the container
  model_path : path to a ggml .bin (fp16 / q8_0 / q4_0 / imatrix-quantized)
"""

from __future__ import annotations

import os
import subprocess
import time
import wave
from pathlib import Path

from engines.base import Engine, Transcription


def _wav_seconds(path: str) -> float:
    """Duration of a 16 kHz mono wav (whisper.cpp's required input format)."""
    with wave.open(path, "rb") as w:
        return w.getnframes() / float(w.getframerate())


class GgmlEngine(Engine):
    name = "ggml"

    def __init__(
        self,
        model_path: str,
        bin_path: str | None = None,
        extra_args: list[str] | None = None,
    ):
        self.model_path = model_path
        self.bin_path = bin_path or os.environ.get("WHISPER_CLI", "whisper-cli")
        self.extra_args = extra_args or []
        if not Path(model_path).exists():
            raise FileNotFoundError(f"ggml model not found: {model_path}")

    def transcribe(self, audio_path: str, lang: str) -> Transcription:
        audio_seconds = _wav_seconds(audio_path)
        cmd = [
            self.bin_path,
            "-m", self.model_path,
            "-f", audio_path,
            "-l", lang,          # force language — no auto-detect, per Loop 1 methods
            "-nt",               # no timestamps in output
            "-otxt", "-of", "-", # text to stdout
            *self.extra_args,
        ]
        start = time.perf_counter()
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True)
        compute_seconds = time.perf_counter() - start
        # -otxt -of - prints the transcript to stdout; stderr carries whisper's own timing log.
        text = proc.stdout.strip()
        return Transcription(text=text, audio_seconds=audio_seconds, compute_seconds=compute_seconds)

    def health(self) -> bool:
        try:
            subprocess.run([self.bin_path, "-h"], capture_output=True, check=True)
            return Path(self.model_path).exists()
        except (subprocess.CalledProcessError, FileNotFoundError):
            return False
