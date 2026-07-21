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


# --- performance note -------------------------------------------------------------------
# The bootstrap must NOT recompute edit distance on every resample: at 10k resamples over
# hundreds of utterances that is millions of Levenshtein calls. Instead we compute each
# utterance's (edits, ref_len) ONCE, then resample over those cached integer pairs and sum.
# The micro-average = sum(edits)/sum(ref_len) is identical to the naive version, just fast.
Counts = list[tuple[int, int]]  # per-utterance (edits, ref_len)


def _edit_counts(pairs: list[tuple[str, str, str]]) -> Counts:
    """Per-utterance (edits, ref_len) after normalization — computed once, reused by every resample."""
    counts: Counts = []
    for hyp, ref, lang in pairs:
        unit = "word" if lang == "en" else "char"
        h = _tokens(normalize(hyp, lang), unit)
        r = _tokens(normalize(ref, lang), unit)
        counts.append((_levenshtein(h, r), len(r)))
    return counts


def _micro(counts: Counts) -> float:
    """Corpus micro-average error = total edits / total reference tokens."""
    total_edits = sum(c[0] for c in counts)
    total_ref = sum(c[1] for c in counts)
    return total_edits / total_ref if total_ref else 0.0


def _corpus_error(pairs: list[tuple[str, str, str]]) -> float:
    """Aggregate error rate = total edits / total reference tokens across utterances.

    Corpus-level micro-average (the standard WER/CER definition), not the mean of per-utterance
    rates — short utterances don't get outsized weight. Kept as a convenience wrapper.
    """
    return _micro(_edit_counts(pairs))


def _percentile_ci(stats: list[float], ci: float) -> tuple[float, float]:
    stats.sort()
    n = len(stats)
    return stats[int((1 - ci) / 2 * n)], stats[int((1 + ci) / 2 * n) - 1]


def bootstrap_error_rate(
    pairs: list[tuple[str, str, str]],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Corpus error rate with a bootstrap CI, resampling over utterances.

    `pairs` is a list of (hypothesis, reference, lang). Edit counts are precomputed once; each
    resample sums cached integer pairs, so 10k resamples on a real corpus run in well under a
    second.
    """
    if not pairs:
        return Estimate(0.0, 0.0, 0.0, 0)
    counts = _edit_counts(pairs)
    point = _micro(counts)
    rng = random.Random(seed)
    n = len(counts)
    stats = []
    for _ in range(n_resamples):
        te = tr = 0
        for _ in range(n):
            e, r = counts[rng.randrange(n)]
            te += e
            tr += r
        stats.append(te / tr if tr else 0.0)
    lo, hi = _percentile_ci(stats, ci)
    return Estimate(point, lo, hi, n)


def bootstrap_difference(
    pairs_a: list[tuple[str, str, str]],
    pairs_b: list[tuple[str, str, str]],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """CI on the difference in corpus error rate (a - b), resampling each set independently.

    The core Loop 1 statistic: e.g. a = Tibetan (Q4_0 minus FP16) degradation, b = English
    degradation. If the returned interval excludes zero, the interaction is supported. Edit
    counts are precomputed once per set.
    """
    if not pairs_a or not pairs_b:
        return Estimate(0.0, 0.0, 0.0, 0)
    ca, cb = _edit_counts(pairs_a), _edit_counts(pairs_b)
    point = _micro(ca) - _micro(cb)
    rng = random.Random(seed)
    na, nb = len(ca), len(cb)
    stats = []
    for _ in range(n_resamples):
        tea = tra = 0
        for _ in range(na):
            e, r = ca[rng.randrange(na)]
            tea += e
            tra += r
        teb = trb = 0
        for _ in range(nb):
            e, r = cb[rng.randrange(nb)]
            teb += e
            trb += r
        stats.append((tea / tra if tra else 0.0) - (teb / trb if trb else 0.0))
    lo, hi = _percentile_ci(stats, ci)
    return Estimate(point, lo, hi, min(na, nb))
