# Preregistration: Quantization × Language-Resource Fairness in Edge ASR
**Project:** From the Bench to the Field — Tibetan/English ASR on edge hardware
**Scope for this cycle:** English and Tibetan throughout; **Sanskrit added to H1 only**
(it is already wired into the harness and corpus loaders, so it rides along at low
marginal cost and gives H3 a third data point). Sanskrit *calibration* (H2) and
mixed-calibration are deferred (see Future Work). Vāksañcayaḥ (Sanskrit) is CC BY-NC 4.0
— non-commercial use must be noted in the paper.
**Status:** 🔒 FROZEN 2026-07-22 — this commit to the public repo locks the document. No
results have been seen at freeze time. Any change to wording, direction, hypotheses,
metrics, or falsifiers after this point is logged as a dated deviation below, not edited
in silently.

---

## Background (condensed)

Whisper's tokenizer treats Tibetan almost as raw bytes — measured at ~15.6× more
decoder steps than English for equivalent content, independent of any quantization.
Naive blockwise quantization (Q4_0/Q8_0-style, e.g. whisper.cpp/GGML) groups weights
into blocks sharing one scale tied to the block's largest value; small weights in a
block dominated by one large weight take ~99% relative error vs. ~5–11% for the large
weight itself — pure arithmetic, no calibration data involved.

Prior work (Ferraz et al., NAVER LABS Europe internship report, 2024 — **the
quantization-bias analysis in that report was not peer-reviewed; only its separate
DistilWhisper method was, at ICASSP 2024**) found that LLM.int8() — a
calibration-aware, activation-outlier-preserving method, not naive rounding — still
amplifies the WER gap for low-resource languages on whisper-large-v2, using
FLEURS/CV-13. Neither Tibetan nor Sanskrit appears in that study; both sit outside its
resourcefulness scale entirely (which bottoms out at 0–10 hours of Whisper training
data — still inside Whisper's 99-language coverage). This is the closest prior work,
and it's evidence for H2 below, not H1 — worth being explicit about, since H1 and H2
test genuinely different mechanisms.

---

## H1 — Naive quantization tracks tokenizer sparsity

> Going from FP16 → Q8_0 → Q4_0 (naive, data-free blockwise quantization, no
> calibration set), error-rate increase will be **ordered by tokenizer byte-fallback
> ratio: English < Sanskrit < Tibetan** — the same ordering already measured on the
> tokenizer, now reproduced downstream by quantization.

- **Mechanism claimed:** arithmetic — small weights sharing a block scale with a large
  weight get crushed; languages relying more on byte-fallback tokenization route more
  through these fragile small-weight paths.
- **Metric:** WER for English; CER for Tibetan and Sanskrit (byte-fallback scripts).
  Degradation reported both in **absolute** points (primary, "user-experienced harm")
  and **relative** percent (secondary, "mechanistic amplification").
- **Prediction (a priori):** the tokenizer probe already measured English < Sanskrit <
  Tibetan; H1 predicts naive-quantization degradation reproduces this ordering, with
  Sanskrit as the middle point.
- **Falsifier:** the degradation ordering is violated — English degrades as much as or
  more than the low-resource languages, or Sanskrit degrades more than Tibetan, despite
  their tokenizer byte-fallback ratios. (Guard: if Q4_0 error exceeds an unusability
  floor for English too, the model is simply broken at 4 bits and there is no gradient
  to attribute — report as inconclusive, not as support.)
- **Status:** mechanism probes (tokenizer ratio measurement across all three languages,
  Q4_0 weight-crushing arithmetic) already run — ready to test end-to-end now.

---

## H2 — Calibration protects English regardless of choice; Tibetan is fragile either way

> **English's WER is robust to calibration-language choice** — calibrating INT8 on
> English or on Tibetan makes little difference to English's WER.
> **Tibetan's WER is fragile to calibration-language choice** — English-calibration
> hurts Tibetan badly, and Tibetan-calibration does not symmetrically rescue it the way
> English-calibration protects English.

**Interpretation:** foundation models compress more dedicated, specialized
representational capacity into English during pretraining. English's representation is
robust enough that it survives regardless of what calibration protects; Tibetan's is
thinner and more dependent on exactly which channels calibration happens to
preserve — a pattern that rhymes with **superposition** in interpretability research
(features packed into few, shared dimensions are more sensitive to perturbation than
features with abundant dedicated dimensions), though this specific mechanism has not
been confirmed for Whisper's audio representations — treat as a motivating analogy,
not a citable fact.

**Design:** 2×2 grid — calibration language (English/Tibetan) × test language
(English/Tibetan). Define:
- Sensitivity(English) = |WER(Eng-calib, test Eng) − WER(Tib-calib, test Eng)|
- Sensitivity(Tibetan) = |WER(Eng-calib, test Tib) − WER(Tib-calib, test Tib)|

**Prediction:** Sensitivity(English) is small; Sensitivity(Tibetan) is large.

- **Falsifier:** Sensitivity(English) ≥ Sensitivity(Tibetan) — i.e., English swings
  around under calibration choice just as much as Tibetan does. This would mean
  English does not have uniquely robust, calibration-independent representational
  capacity, and the story above is wrong or unsupported by this experiment.
- **Real-world check:** compare both calibration conditions against the existing
  fine-tuned checkpoint (`milanakdj/whisper-small-full-tibetan`). If fine-tuning closes
  Tibetan's gap far more than either calibration condition does, that supports "this is
  representational, not fixable by calibration alone" — and motivates fine-tuning
  (QLoRA or full fine-tuning) as the actual fix, left as future work for this cycle.
- **Status:** not yet run — needs English-calibrated and Tibetan-calibrated INT8
  builds, both tested on both languages.

---

## H3 — Tokenizer compression predicts both H1 and H2 (unifying claim)

> Tokenizer compression ratio (byte-fallback ratio) is the single upstream signal that
> predicts *both* naive-quantization fragility (H1) *and* calibration-asymmetry (H2) —
> one root cause (how much dedicated representational capacity a language received
> during pretraining, visible via tokenizer compression) explains two downstream
> symptoms.

- **Falsifier:** H1 holds (naive quantization damage tracks tokenizer ratio) but H2's
  asymmetric-sensitivity pattern does not appear, or vice versa — the two phenomena
  decouple, meaning they have different root causes that merely looked similar on the
  surface.
- **Critical caveat — state this plainly in the writeup, do not oversell:** H1 now
  spans **three languages** (English, Sanskrit, Tibetan), so its *ordering* claim is a
  genuine 3-point test rather than a single contrast. But H2 (calibration asymmetry) is
  measured on **English and Tibetan only**, so the H1↔H2 *linkage* — the claim that one
  upstream signal drives both symptoms — still rests on N=2 for the two low-resource
  languages that have both measurements. Report the 3-point H1 ordering as suggestive
  evidence for the shared root cause, but report the unifying H3 claim itself as a
  hypothesis worth testing further (Sanskrit calibration would close this — deferred,
  see Future Work), not as a confirmed predictive relationship.

---

## Metrics (preregistered)

The study spans four measurement problems. Metrics are marked **[primary]** (locked —
reported regardless of outcome) or **[secondary]** (reported if time allows; a null or
absence is not a failure). Prereg discipline: we do not imply we ran a metric we did
not, and we commit to the primary set now.

**1. Core ASR accuracy** — the H1/H2 outcome variables.
- **[primary] WER** — (S + D + I) / reference words, via `jiwer`. English's headline number.
- **[primary] CER** — character-level error. **Reported alongside WER for every
  language, and treated as the more honest primary for Tibetan**, which is not
  whitespace-delimited so word-level WER is shaky for it. Sanskrit likewise reported in CER.
- **[primary] Normalizer disclosure** — the exact normalization/tokenization path is
  documented (see `bench/normalize.py`); the OpenWER result of up to 25-point WER swings
  by normalizer choice means the number is meaningless without it. One normalizer, stated.

**2. Edge / efficiency** — the "edge" half of the paper.
- **[primary] RTF** — compute ÷ audio duration; < 1.0 is the deployability threshold.
- **[primary] Peak memory (RAM/VRAM)** and **power draw (W)** — via `tegrastats`, which
  the runner already wraps around every sweep. Power is the metric most quantization
  papers skip — a genuine differentiator for the edge framing.
- **[primary] Model size on disk (MB)** — concrete "can this fit on the device" number.
- **[secondary] Load time** — model load separate from inference; matters for real
  deployment, usually ignored.

**3. Fairness / bias** — operationalizes what H1/H2 test the *cause* of.
- **[primary] Relative degradation (%)** — (WER_quant − WER_base) / WER_base — the
  cross-language comparison metric, since Tibetan (~45%) and English (~6%) baselines make
  the same absolute +5 points mean very different things. Absolute point change reported too.
- **[primary] Min-max gap** — best minus worst language; the single number a reviewer
  reads as "fairness," and exactly the quantity H1/H2 explain the cause of.
- **[secondary] Macro-average WER/CER** — averaged across languages, not utterances, so
  high-resource languages don't dominate. (With 2–3 languages this is nearly the raw mean;
  reported for completeness.)
- **[secondary] Per-sentence degradation histograms** — bucket clips into got-worse /
  same / got-better after quantization (Ferraz et al.'s technique), rather than one average.
  Aligns with the paired-comparison reporting plan below.

**4. Meaning-preservation / translation quality** — **Phase 3, out of scope for the
Aug 10 poster's H1/H2 core; specified here so the plan is fixed before that phase.**
Per prior discussion, when Phase 3 runs: **lead with [primary] key-point/claim recall**
(QAGS/QuestEval-style — the closest operationalization of "did the main signal survive")
**and [primary] COMET-KIWI** (learned MT quality, better human-correlated than BLEU or raw
cosine); **[secondary] cosine similarity** kept only as the cheap triage filter (demoted
from primary); **[secondary] NLI/entailment** for negation flips and dropped qualifiers;
BLEU reported only if a reviewer expects it. Pick the two most informative and say why —
do not run all five.

**5. Quantization-specific fidelity** — supports the mechanism claims, not WER-shaped.
- **[primary] Weight/activation error** — relative error per quantization block, the
  Q4_0 weight-crushing arithmetic **already measured** (`findings/loop1_quant_mechanism.md`).
  This is the mechanistic evidence, distinct from any downstream accuracy metric.
- **[secondary] KL divergence** between quantized and FP16 output distributions, and
  **perplexity ratio** (quant vs. FP16) — direct measures of prediction shift independent
  of whether WER caught it. Reported only if logit access via the engine is tractable in time.

**Timeline-honest primary set:** WER + CER + RTF + peak-memory/power/size + relative
degradation + min-max gap covers Phase 1 (H1/H2) solidly. Everything above marked
[secondary], and all of Phase 3, is run only if the budget allows — stated as such in methods.

## Priority order given the Aug 10 deadline

1. **H1** — freeze and run now, across all three languages (English, Sanskrit,
   Tibetan). Mechanism probes already done; most tractable. Sanskrit rides along at low
   marginal cost because the corpus loader and model ladder already support it.
2. **H2** — run if time allows. English×Tibetan only. Needs two calibrated INT8 builds;
   the more novel contribution, and directly testable against the existing fine-tuned
   checkpoint.
3. **H3** — report as an observation across H1+H2 results. H1's 3-point ordering is a
   real test; the H1↔H2 linkage is still N=2. Not a separately-run experiment.

## Future work (explicitly deferred, not silently dropped)

- **Sanskrit calibration (H2)** — a Sanskrit calibration cache + 3×3 grid would let the
  H1↔H2 linkage (H3) be tested at three points instead of two; deferred as too costly
  this cycle.
- **Mixed/multilingual calibration** — the constructive follow-up to H2: does
  including Tibetan in a broader multilingual calibration set narrow the gap, even if
  Tibetan-only calibration doesn't?
- **Fine-tuning comparison (QLoRA / full fine-tuning)** — the real fix implied if H2
  holds; only a preliminary checkpoint comparison is in scope this cycle, not a full
  fine-tuning experiment of your own.

## Small-N reporting plan (applies throughout)

- Report **ranges/bootstrap confidence intervals**, not single point WER estimates.
- Use **paired comparisons** — same clips across all quantization/calibration
  conditions, not independently-sampled groups.
- Report **effect size and per-clip consistency** over p-values.
- State exact sample size and the reason for it plainly in methods — no implying more
  statistical power than the data supports.
