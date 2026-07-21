"""Versioned data contracts shared across every stage of the pipeline.

These are the single source of truth for payload shapes. Every module that produces
or consumes a benchmark result imports from here rather than passing raw dicts, so a
schema change is a one-line diff that shows up in review. The microservice split in
`docs/roadmap.md` (Loop 5) will import these unchanged.
"""

from schemas.transcript_row import TranscriptRow
from schemas.segment import Segment
from schemas.score_row import ScoreRow

SCHEMA_VERSION = "0.1.0"

__all__ = ["TranscriptRow", "Segment", "ScoreRow", "SCHEMA_VERSION"]
