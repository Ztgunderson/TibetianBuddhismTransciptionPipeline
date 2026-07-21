# Preregistration — Loop 1 (Quantization Tax)

**Status:** 🟡 DRAFT — awaiting sign-off on Decisions 1 & 2 below. Nothing in the full Loop 1
grid runs until this file is committed with a real date and `Status: FROZEN`.

**Purpose.** Fix the hypotheses, the analysis, and the interpretation rules *before* any results
exist, so the finding is confirmatory rather than post-hoc. Amending after seeing results is
allowed but must be recorded as an amendment with its own date, leaving the original intact.

---

## Hypothesis

**H1.** Accuracy degradation from FP16 → Q4_0 is disproportionately larger for low-resource
languages (Tibetan, Sanskrit) than for high-resource English — a significant language ×
quantization *interaction*, not merely a main effect of either.

**Predicted direction:** degradation(Tibetan) > degradation(Sanskrit) > degradation(English),
reflecting decreasing scarcity in Whisper's training distribution.

---

## Design (fixed)

- **Models:** whisper-large-v3, whisper-small.
- **Quant levels:** FP16 (baseline), Q8_0, Q4_0. (Engine A / ggml. Q5_0 intentionally excluded.)
- **Languages/corpora:** English/FLEURS, Sanskrit/Vāksañcayaḥ, Tibetan/NICT-Tib1.
- **Sample:** ~30 min audio per cell (≈200–400 utterances), seeded sampling
  (`seed=0`), speaker-stratified where the corpus allows.
- **Metrics:** WER (English), CER (Tibetan, Sanskrit), RTF, watts/MB (tegrastats).
- **Primary statistic:** bootstrap CI (10k resamples over utterances) on the interaction
  contrast — see below.
- **Hardware:** Jetson AGX Orin, clocks locked (`jetson_clocks`) before every run.

**Degradation** for a (language, model) is defined as the change in error rate from FP16 to
Q4_0. The **interaction contrast** is:

    Δ_interaction = [err(lang_LR, Q4_0) − err(lang_LR, FP16)]
                  − [err(English, Q4_0) − err(English, FP16)]

computed per low-resource language, with a bootstrap CI via
`bench.metrics.bootstrap_difference`. **H1 is supported for that language iff the CI excludes 0
in the predicted (positive) direction.**

---

## Decision 1 — What "regressive" means  ⬜ NEEDS SIGN-OFF

Absolute and relative framings disagree (Tibetan 40→60% CER and English 5→7.5% WER is +20 vs
+2.5 points, but +50% for both). **Proposed default (recommended):** preregister *both*, reported
under distinct names, neither allowed to silently stand in for the other:

- **User-experienced harm** = the *absolute* point increase in error rate. Rationale: 20 points
  of CER is an unusable transcript regardless of starting point. This is the primary poster claim.
- **Mechanistic amplification** = the *relative* (proportional) increase. The stricter test; may
  fail even if absolute harm is large.

The interaction contrast above is stated in **absolute** terms (user-experienced harm) as
primary; the relative version is reported alongside as secondary.

> **Your call:** accept this both-metrics default, or pick a single metric. Sign here: __________

## Decision 2 — The falsifier  ⬜ NEEDS SIGN-OFF

H1 is **abandoned** (not quietly reframed) if any of the following holds:

- **Proposed default (recommended):** the absolute interaction contrast CI includes 0 for *both*
  low-resource languages — i.e. no detectable disproportionate absolute harm.
- *Optionally also:* a **floor effect** — Q4_0 error exceeds a pre-set unusability threshold
  (proposed: WER/CER > 0.80) for *English too*, meaning the model is simply broken at 4 bits and
  there is no gradient to attribute to representation.

> **Your call:** accept this falsifier, adjust the threshold, or specify your own. Sign here: ______

---

## What will be reported regardless of outcome

- Every cell's error rate + CI, RTF, and watts/MB, including cells where H1 fails.
- If H1 fails: the null result is published as-is (and Loop 2's rationale is revisited).
- The domain-mismatch limitation (studio read speech vs. dharma audio) stated prominently.
- The exact sampled manifests (committed under `data/results/{experiment_id}/`) so any figure
  regenerates from an exact past run.

---

## Sign-off

| Field | Value |
|---|---|
| Decisions 1 & 2 accepted by | ztgunderson — __________ |
| Date frozen | __________ |
| Status | 🟡 DRAFT (change to FROZEN on sign-off) |
