"""A language/speaker turn within a recording — the output of the segmentation stage.

Not exercised in the Loop 0-2 sprint (benchmarks feed pre-segmented utterances straight
to ASR). Defined now so the segmentation service in `docs/roadmap.md` (Loop 4/5) imports
this contract rather than inventing its own.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Segment(BaseModel):
    start_s: float = Field(..., description="Segment start, seconds from file start")
    end_s: float = Field(..., description="Segment end, seconds from file start")
    language: str = Field(..., description="ISO code where possible, else 'unknown'")
    speaker: str | None = Field(None, description="Speaker label if diarization ran")
    method: str = Field(..., description="'mms_lid' | 'voxlect_dialect' | 'diarization' | 'whisper_auto'")
    confidence: float | None = Field(None, description="LID/diarization confidence if the model exposes one")
    schema_version: str = Field("0.1.0")
