# Loop 1 — The Quantization Tax (conference core)

**Subtitle:** "The Bench" — controlled conditions, ground-truth labels, the ideal case under
which the gap between languages should be *smallest*.

## Hypothesis

> **H1.** Accuracy degradation from FP16 → Q4_0 is disproportionately larger for low-resource
> languages (Tibetan, Sanskrit) than for high-resource English — a significant language ×
> quantization *interaction*, not merely a main effect of either.

### Mechanism

Block-wise quantization computes one scale factor per ~32 weights and preserves what is large
relative to that scale. Weights repeatedly reinforced by abundant training data are
high-magnitude and survive; representations for rare languages are lower-magnitude and sparsely
activated, so they are preferentially crushed. If this is right, damage is *predicted by corpus
representation* — which the dialect extension (below) and Loop 2 test directly.

This makes concrete the survey's (arXiv 2510.19144) unmeasured claim of "token-level sparsity
and structural underrepresentation."

## Design

3 languages × 3 quant levels (FP16, Q8_0, Q4_0) × 2 models (whisper-large-v3, whisper-small)
= 18 cells. ~30 min of audio per cell (≈200–400 utterances) — enough for bootstrap CIs tight
enough to detect a moderate interaction, and the grid finishes overnight on the Orin.
`sudo jetson_clocks` locked before every run.

- **Metrics:** WER (English), CER (Tibetan, Sanskrit), RTF, plus measured watts/MB from
  `tegrastats`.
- **Statistics:** bootstrap CIs (10k resamples over utterances). The interaction is tested with
  `bench.metrics.bootstrap_difference` on degradation deltas — if the CI on
  (Tibetan Q4−FP16) − (English Q4−FP16) excludes zero, H1 is supported.
- **Dialect extension (stretch):** add TIBMD@MUC's Ü-Tsang / Amdo / Kham. Same script, same
  family, same corpus — isolates *representation* from "Tibetan is acoustically hard," and
  predicts an ordering (Ü-Tsang < Amdo < Kham in degradation), not just a difference.

## Loop 1b — token-level sparsity probe (1 day, alongside)

Measure tokens emitted per second of audio, per language, under Whisper's BPE. Tibetan script
likely triggers byte-fallback, inflating token count and multiplying the autoregressive steps
at which errors compound. A 3× tokens/sec gap is both a mechanistic explanation for H1 and a
directly citable extension of the survey.

## Preregistration (before any full run)

Two decisions are frozen in `PREREGISTRATION.md`, committed and timestamped *before* results
exist — this is what makes the finding confirmatory rather than post-hoc:

1. **What "regressive" means.** Absolute vs. relative degradation disagree (Tibetan 40→60% CER
   and English 5→7.5% WER is +20 vs +2.5 points but +50% for both). Both are pre-registered
   under distinct names — *user-experienced harm* (absolute) and *mechanistic amplification*
   (relative) — stating in advance which supports which claim.
2. **The falsifier.** The result that makes H1 abandoned (e.g. proportional degradation with a
   non-significant interaction; or a Q4_0 floor effect that breaks every language equally).

## Gate

An interaction effect whose CI excludes zero. If H1 fails, that is publishable and a more
interesting post than confirmation — but Loop 2's rationale weakens and it is redesigned.

## Limitations (for the report)

Studio read speech, not spontaneous dharma teaching (domain mismatch — the honest Tibetan-corpus
story; see Loop 0). Single hardware platform. No Tibetan spontaneous-speech benchmark with
character-level ground truth exists at scale, and the survey's de facto benchmark (IARPA Babel)
is LDC-licensed and not freely obtainable — an accessibility barrier worth stating.
