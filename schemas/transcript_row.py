"""One transcription result: a single (utterance, model, quant, engine) cell of the grid.

This is the atomic unit the Loop 1/1b/2 benchmark writes and the analysis reads. One row
per utterance per configuration, so bootstrap CIs can resample over `utterance_id`.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class TranscriptRow(BaseModel):
    """A single decoded utterance under one model configuration.

    Error metrics (`wer` / `cer`) are only populated when ground-truth reference text
    exists (all of Loop 1 — every corpus here ships references). On real Deer Park audio
    (roadmap Loop 5) they stay `None` and divergence is measured instead.
    """

    # --- identity: what was decoded, under what configuration ---
    utterance_id: str = Field(..., description="Stable per-utterance id from the corpus manifest")
    corpus: str = Field(..., description="Corpus key, e.g. 'fleurs' | 'nict_tib1' | 'vaksancayah' | 'tibmd'")
    lang: str = Field(..., description="ISO code: 'en' | 'bo' | 'sa'")
    dialect: str | None = Field(None, description="For Tibetan: 'u-tsang' | 'kham' | 'amdo'; else None")
    speaker_id: str | None = Field(None, description="Speaker id where the corpus provides one")

    engine: str = Field(..., description="'ggml' (whisper.cpp) | 'ct2' (faster-whisper/CTranslate2)")
    model: str = Field(..., description="'whisper-large-v3' | 'whisper-small'")
    quant: str = Field(..., description="'fp16' | 'q8_0' | 'q4_0' | 'int8'")
    calibration: str | None = Field(
        None,
        description="Loop 2 only: calibration set used for imatrix/INT8, e.g. 'generic' | 'en' | 'bo'",
    )

    # --- outputs ---
    hypothesis: str = Field(..., description="Model output text (normalized form stored separately)")
    reference: str | None = Field(None, description="Ground-truth text when available")

    # --- metrics (per-utterance; aggregated with CIs downstream) ---
    wer: float | None = Field(None, description="Word error rate; set for English")
    cer: float | None = Field(None, description="Character error rate; set for Tibetan/Sanskrit")
    rtf: float = Field(..., description="Real-time factor = compute_time / audio_duration; <1 is faster than realtime")

    # --- provenance ---
    audio_seconds: float = Field(..., description="Utterance duration in seconds")
    compute_seconds: float = Field(..., description="Wall-clock decode time in seconds")
    experiment_id: str = Field(..., description="Run id; groups a full sweep for reproducibility")
    schema_version: str = Field("0.1.0", description="Bump on any breaking field change")
