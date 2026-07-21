"""One meaning-preservation score, one method, one segment pair.

Not exercised in the Loop 0-2 sprint. Defined now for the five-method scoring service in
`docs/roadmap.md` (Loop 6), which runs methods side by side and treats their divergence as
the finding — hence one row per method rather than a pre-combined aggregate.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ScoreRow(BaseModel):
    pair_id: str = Field(..., description="Identifies the (source, candidate-translation) pair")
    method: str = Field(..., description="'cosine' | 'key_point' | 'comet_kiwi' | 'roundtrip' | 'nli'")
    score: float = Field(..., description="Normalized 0-1 within the method's own scale")
    detail: dict = Field(default_factory=dict, description="Method-specific extra info, e.g. which key points failed")
    schema_version: str = Field("0.1.0")
