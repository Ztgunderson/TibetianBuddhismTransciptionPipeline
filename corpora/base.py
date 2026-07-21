"""The uniform corpus manifest and deterministic sampling.

Every loader, regardless of the corpus's native on-disk format (Kaldi wav.scp, HF parquet,
plain tsv), converts to a list of `Utterance`. The runner only ever sees this shape.
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class Utterance:
    """One audio clip with its ground-truth reference and metadata."""

    utterance_id: str
    audio_path: str          # absolute path to a 16 kHz mono wav (resampled at load time)
    reference: str           # ground-truth transcript in native script
    lang: str                # 'en' | 'bo' | 'sa'
    corpus: str
    dialect: str | None = None      # Tibetan only: 'u-tsang' | 'kham' | 'amdo'
    speaker_id: str | None = None
    audio_seconds: float | None = None


Manifest = list[Utterance]


def sample_manifest(
    manifest: Manifest,
    minutes: float | None = None,
    max_utterances: int | None = None,
    seed: int = 0,
    stratify_by: str | None = None,
) -> Manifest:
    """Deterministically sample a manifest down to a target budget.

    minutes        — approximate total audio to keep (requires audio_seconds populated)
    max_utterances — hard cap on count
    seed           — fixed for reproducibility; same seed + same manifest => same subset
    stratify_by    — attribute name ('speaker_id' | 'dialect') to spread the sample across,
                     so a 30-minute Tibetan sample isn't accidentally one speaker

    Selection order is a seeded shuffle, so the result is stable and documented by (seed,
    budget) alone.
    """
    rng = random.Random(seed)
    items = list(manifest)

    if stratify_by:
        buckets: dict = {}
        for u in items:
            buckets.setdefault(getattr(u, stratify_by), []).append(u)
        for b in buckets.values():
            rng.shuffle(b)
        # round-robin across buckets for an even spread
        ordered: Manifest = []
        while any(buckets.values()):
            for key in list(buckets):
                if buckets[key]:
                    ordered.append(buckets[key].pop())
                if not buckets[key]:
                    del buckets[key]
        items = ordered
    else:
        rng.shuffle(items)

    selected: Manifest = []
    total_seconds = 0.0
    for u in items:
        if max_utterances is not None and len(selected) >= max_utterances:
            break
        if minutes is not None and total_seconds >= minutes * 60:
            break
        selected.append(u)
        total_seconds += u.audio_seconds or 0.0
    return selected


def write_manifest(manifest: Manifest, path: str | Path) -> None:
    """Persist a sampled manifest so an exact run can be regenerated from disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for u in manifest:
            f.write(json.dumps(asdict(u), ensure_ascii=False) + "\n")


def read_manifest(path: str | Path) -> Manifest:
    """Load a manifest previously written by `write_manifest`."""
    out: Manifest = []
    with Path(path).open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(Utterance(**json.loads(line)))
    return out
