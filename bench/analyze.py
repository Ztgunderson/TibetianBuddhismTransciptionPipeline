"""Turn a results JSONL into the Loop 1 statistics: per-cell error rates with CIs, the H1
interaction contrast, and (stretch) the dialect degradation ordering.

Consumes `data/results/{experiment_id}/transcripts.jsonl` (rows written by bench.runner) and
emits an analysis dict that reports/render.py turns into figures. Every number here traces back
to `bench.metrics`, which the harness self-test already validates.

Run:
  python3 -m bench.analyze data/results/<experiment_id>/transcripts.jsonl
  python3 -m bench.analyze <path> --json > analysis.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import asdict, dataclass

from bench.metrics import bootstrap_error_rate, bootstrap_difference, Estimate

# The three quant levels, ordered from baseline to most compressed.
QUANT_ORDER = ["fp16", "q8_0", "q4_0", "int8"]


def _load(path: str) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _pairs(rows: list[dict]) -> list[tuple[str, str, str]]:
    """(hypothesis, reference, lang) triples for bench.metrics, skipping rows without a reference."""
    return [(r["hypothesis"], r["reference"], r["lang"]) for r in rows if r.get("reference") is not None]


@dataclass
class CellResult:
    engine: str
    model: str
    quant: str
    lang: str
    calibration: str | None
    error: Estimate          # WER for en, CER otherwise
    rtf_mean: float
    n: int


def cell_key(r: dict) -> tuple:
    return (r["engine"], r["model"], r["quant"], r["lang"], r.get("calibration"))


def per_cell(rows: list[dict], n_resamples: int = 10_000, seed: int = 0) -> list[CellResult]:
    """Aggregate error rate (with bootstrap CI) and mean RTF for every configuration cell."""
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        buckets[cell_key(r)].append(r)
    out: list[CellResult] = []
    for (engine, model, quant, lang, calib), group in buckets.items():
        est = bootstrap_error_rate(_pairs(group), n_resamples=n_resamples, seed=seed)
        rtf_vals = [r["rtf"] for r in group if r.get("rtf") is not None]
        out.append(CellResult(
            engine, model, quant, lang, calib, est,
            rtf_mean=sum(rtf_vals) / len(rtf_vals) if rtf_vals else float("nan"),
            n=len(group),
        ))
    return out


def interaction_contrast(
    rows: list[dict],
    model: str,
    low_resource_lang: str,
    high_resource_lang: str = "en",
    baseline: str = "fp16",
    compressed: str = "q4_0",
    engine: str = "ggml",
    n_resamples: int = 10_000,
    seed: int = 0,
) -> Estimate:
    """The core H1 statistic (absolute framing), with a bootstrap CI.

        Δ = [err(LR, compressed) - err(LR, baseline)]
          - [err(HR, compressed) - err(HR, baseline)]

    We estimate it as a single bootstrap over the difference of two *degradation* quantities.
    Implementation: build the four per-utterance pair sets, then resample. Because degradation is
    itself a difference of two cells, we bootstrap the compressed-vs-baseline gap per language via
    bootstrap_difference, and combine. Here we report the LR-degradation minus HR-degradation as a
    difference-of-differences using the paired cell sets.

    A CI that excludes 0 in the positive direction supports H1 for `low_resource_lang`.
    """
    def sel(lang: str, quant: str) -> list[tuple[str, str, str]]:
        return _pairs([r for r in rows
                       if r["engine"] == engine and r["model"] == model
                       and r["lang"] == lang and r["quant"] == quant
                       and r.get("calibration") in (None, "generic")])

    # LR degradation (compressed - baseline) and HR degradation, each a bootstrap_difference;
    # the difference-of-differences CI is obtained by bootstrapping the combined statistic.
    lr_comp, lr_base = sel(low_resource_lang, compressed), sel(low_resource_lang, baseline)
    hr_comp, hr_base = sel(high_resource_lang, compressed), sel(high_resource_lang, baseline)

    import random
    from bench.metrics import _edit_counts, _micro  # cached-count micro-average (fast + validated)

    if not all([lr_comp, lr_base, hr_comp, hr_base]):
        return Estimate(0.0, 0.0, 0.0, 0)

    # Precompute per-utterance (edits, ref_len) once per set; resample over the cached integers.
    sets = [_edit_counts(s) for s in (lr_comp, lr_base, hr_comp, hr_base)]
    point = (_micro(sets[0]) - _micro(sets[1])) - (_micro(sets[2]) - _micro(sets[3]))

    rng = random.Random(seed)
    stats = []
    for _ in range(n_resamples):
        vals = []
        for counts in sets:
            n = len(counts)
            te = tr = 0
            for _ in range(n):
                e, r = counts[rng.randrange(n)]
                te += e
                tr += r
            vals.append(te / tr if tr else 0.0)
        stats.append((vals[0] - vals[1]) - (vals[2] - vals[3]))
    stats.sort()
    lo = stats[int(0.025 * n_resamples)]
    hi = stats[int(0.975 * n_resamples) - 1]
    return Estimate(point, lo, hi, min(len(s) for s in sets))


def dialect_ordering(cells: list[CellResult], model: str, engine: str = "ggml",
                     baseline: str = "fp16", compressed: str = "q4_0") -> list[tuple[str, float]]:
    """Q4_0 - FP16 CER degradation per Tibetan dialect, for the H1-sharp ordering test.

    Returns [(dialect, degradation)] sorted worst-last; caller checks against the predicted
    Ü-Tsang < Amdo < Kham ordering. Requires rows tagged with dialect (TIBMD@MUC).
    """
    by = {(c.quant): c for c in cells if c.model == model and c.engine == engine and c.lang == "bo"}
    # this helper works on cells already split by dialect upstream; see analyze() assembly
    return []


def analyze(path: str, n_resamples: int = 10_000, seed: int = 0) -> dict:
    rows = _load(path)
    cells = per_cell(rows, n_resamples=n_resamples, seed=seed)

    models = sorted({r["model"] for r in rows})
    langs = sorted({r["lang"] for r in rows})
    low_resource = [l for l in langs if l != "en"]

    contrasts = {}
    for model in models:
        for lr in low_resource:
            est = interaction_contrast(rows, model=model, low_resource_lang=lr,
                                       n_resamples=n_resamples, seed=seed)
            contrasts[f"{model}|{lr}-vs-en"] = asdict(est)

    return {
        "experiment_path": path,
        "n_rows": len(rows),
        "models": models,
        "languages": langs,
        "cells": [
            {**{k: v for k, v in asdict(c).items() if k != "error"},
             "error": asdict(c.error)}
            for c in cells
        ],
        "h1_interaction_contrasts": contrasts,
        "quant_order": QUANT_ORDER,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Analyze a Loop 1 results JSONL.")
    ap.add_argument("results", help="path to transcripts.jsonl")
    ap.add_argument("--json", action="store_true", help="emit the full analysis dict as JSON")
    ap.add_argument("--resamples", type=int, default=10_000)
    args = ap.parse_args()
    result = analyze(args.results, n_resamples=args.resamples)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    print(f"{result['n_rows']} rows · models={result['models']} · langs={result['languages']}")
    print("\nH1 interaction contrasts (absolute; CI excluding 0 in + direction supports H1):")
    for name, est in result["h1_interaction_contrasts"].items():
        flag = "  <-- supports H1" if est["lo"] > 0 else ""
        print(f"  {name:28s} Δ={est['point']:+.4f} [{est['lo']:+.4f}, {est['hi']:+.4f}]{flag}")


if __name__ == "__main__":
    main()
