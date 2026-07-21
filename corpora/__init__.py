"""Corpus loaders. Each emits a uniform manifest of Utterance records so the benchmark
runner is corpus-agnostic. Deterministic, seeded sampling means a run reproduces from the
manifest alone.

Registered corpora (see docs/loop0_harness.md for licenses and download):
  fleurs       — English baseline (FLEURS), CC BY 4.0
  vaksancayah  — Sanskrit (IIT Bombay), CC BY-NC 4.0
  nict_tib1    — Tibetan, Lhasa read speech (OpenSLR 158), CC BY 4.0
  tibmd        — Tibetan dialects Ü-Tsang/Kham/Amdo (OpenSLR 124), CC BY-SA 4.0
"""

from corpora.base import Utterance, Manifest, sample_manifest
from corpora.registry import load_corpus, CORPORA

__all__ = ["Utterance", "Manifest", "sample_manifest", "load_corpus", "CORPORA"]
