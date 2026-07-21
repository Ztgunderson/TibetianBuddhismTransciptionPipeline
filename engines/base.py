"""The engine contract every backend implements, plus a registry.

An Engine takes an audio file path and a forced language, and returns text plus the timing
needed to compute RTF. It does NOT know about metrics, corpora, or the sweep — it only
transcribes. This keeps the Jetson-vs-engine comparisons in the study meaningful: swapping
whisper.cpp for CTranslate2 is a config change, not a code change.
"""

from __future__ import annotations

import abc
from dataclasses import dataclass


@dataclass
class Transcription:
    """One decode result with the timing RTF needs."""

    text: str
    audio_seconds: float
    compute_seconds: float


class Engine(abc.ABC):
    """Base class for an ASR backend at a fixed (model, quant, calibration) configuration.

    A concrete Engine is constructed already bound to one model file / quant level, so the
    runner instantiates one Engine per grid cell and calls `transcribe` per utterance.
    """

    #: short id stored in TranscriptRow.engine, e.g. "ggml" | "ct2"
    name: str = "base"

    @abc.abstractmethod
    def transcribe(self, audio_path: str, lang: str) -> Transcription:
        """Decode one audio file with the language forced. `lang` is an ISO code (en/bo/sa)."""

    def health(self) -> bool:
        """Cheap readiness check (model loaded, GPU visible). Override in subclasses."""
        return True

    def close(self) -> None:
        """Release model/GPU resources. Override where needed."""


# --------------------------------------------------------------------------- registry
def get_engine(name: str, **cfg) -> Engine:
    """Construct an engine by short name. Imports lazily so a missing optional dependency
    (e.g. CTranslate2 before Engine B is built) never breaks Engine A runs."""
    if name == "ggml":
        from engines.ggml import GgmlEngine

        return GgmlEngine(**cfg)
    if name == "ct2":
        from engines.ct2 import Ct2Engine

        return Ct2Engine(**cfg)
    raise ValueError(f"unknown engine: {name!r} (expected 'ggml' or 'ct2')")
