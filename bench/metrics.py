"""Error metrics (WER/CER), real-time factor, and bootstrap confidence intervals.

WER is word-tokenized (English). CER is character-tokenized (Tibetan/Sanskrit and any
non-Latin script) — the standard choice for agglutinative morphology and scripts where
"word" boundaries are ill-defined. Both are computed via Levenshtein edit distance over the
appropriate token stream, after `bench.normalize`.

The bootstrap here resamples over UTTERANCES, not over characters/words: the unit of
independence is the utterance, and CIs must reflect that. This is what lets Loop 1 claim an
interaction with an interval that excludes zero (or honestly report that it doesn't).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from bench.normalize import normalize


# --------------------------------------------------------------------------- edit distance
def _levenshtein(a: list, b: list) -> int:
    """Classic O(len(a)*len(b)) edit distance over token sequences."""
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _tokens(text: str, unit: str) -> list:
    return list(text) if unit == "char" else text.split()


def error_rate(hypothesis: str, reference: str, lang: str) -> float:
    """WER for English, CER otherwise. Returns edits / reference-length (0.0 if ref empty)."""
    unit = "word" if lang == "en" else "char"
    hyp = _tokens(normalize(hypothesis, lang), unit)
    ref = _tokens(normalize(reference, lang), unit)
    if not ref:
        return 0.0
    return _levenshtein(hyp, ref) / len(ref)


def rtf(compute_seconds: float, audio_seconds: float) -> float:
    """Real-time factor. <1.0 is faster than realtime. Guards against zero-length audio."""
    return compute_seconds / audio_seconds if audio_seconds > 0 else float("inf")


# --------------------------------------------------------------------------- aggregation
@dataclass
class Estimate:
    """A point estimate with a bootstrap percentile CI."""

    point: float
    lo: float
    hi: float
    n: int

    def __str__(self) -> str:
        return f"{self.point:.4f} [{self.lo:.4f}, {self.hi:.4f}] (n={self.n})"


def _corpus_error(pairs: list[tuple[str, str, str]]) -> float:
    """Aggregate error rate = total edits / total reference tokens across utterances.

    This is the corpus-level micro-average (the standard WER/CER definition), not the mean of
    per-utterance rates — short utterances don't get outsized weight.
    """
    total_edits = 0
    total_ref = 0
    for hyp, ref, lang in pairs:
        unit = "word" if lang == "en" else "char"
        h = _tokens(normalize(hyp, lang), unit)
        r = _tokens(normalize(ref, lang), unit)
        total_edits += _levenshtein(h, r)
        total_ref += len(r)
    return total_edits / total_ref if total_ref else 0.0


def bootstrap_error_rate(
    pairs: list[tuple[str, str, str]],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Corpus error rate with a bootstrap CI, resampling over utterances.

    `pairs` is a list of (hypothesis, reference, lang). Returns the point estimate on the
    full set plus percentile CI bounds from `n_resamples` resamples-with-replacement.
    """
    if not pairs:
        return Estimate(0.0, 0.0, 0.0, 0)
    point = _corpus_error(pairs)
    rng = random.Random(seed)
    n = len(pairs)
    stats = []
    for _ in range(n_resamples):
        sample = [pairs[rng.randrange(n)] for _ in range(n)]
        stats.append(_corpus_error(sample))
    stats.sort()
    lo_idx = int((1 - ci) / 2 * n_resamples)
    hi_idx = int((1 + ci) / 2 * n_resamples) - 1
    return Estimate(point, stats[lo_idx], stats[hi_idx], n)


def bootstrap_difference(
    pairs_a: list[tuple[str, str, str]],
    pairs_b: list[tuple[str, str, str]],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """CI on the difference in corpus error rate (a - b), resampling each set independently.

    This is the core Loop 1 statistic: e.g. a = Tibetan (Q4_0 minus FP16) degradation, b =
    English degradation. If the returned interval excludes zero, the interaction is supported.
    Caller passes already-differenced inputs, or uses this directly on two cells.
    """
    if not pairs_a or not pairs_b:
        return Estimate(0.0, 0.0, 0.0, 0)
    point = _corpus_error(pairs_a) - _corpus_error(pairs_b)
    rng = random.Random(seed)
    na, nb = len(pairs_a), len(pairs_b)
    stats = []
    for _ in range(n_resamples):
        sa = [pairs_a[rng.randrange(na)] for _ in range(na)]
        sb = [pairs_b[rng.randrange(nb)] for _ in range(nb)]
        stats.append(_corpus_error(sa) - _corpus_error(sb))
    stats.sort()
    lo_idx = int((1 - ci) / 2 * n_resamples)
    hi_idx = int((1 + ci) / 2 * n_resamples) - 1
    return Estimate(point, stats[lo_idx], stats[hi_idx], min(na, nb))
