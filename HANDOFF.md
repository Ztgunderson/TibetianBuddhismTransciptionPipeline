# Where We Left Off — Decide Your Poster Hypothesis

**Last worked:** 2026-07-21 · **Your next job:** review the findings, pick H1's wording + falsifier,
then freeze `PREREGISTRATION.md`. Nothing in the full Loop 1 grid runs until you do.

---

## Read these three first (10 min)

1. `findings/loop1b_tokenizer_probe.md` — Tibetan costs 15.6× more decoder tokens than English;
   the ordering (English < Sanskrit < Tibetan) already matches H1's prediction, with no
   quantization involved. Rerun anytime: `python3 -m bench.tokenizer_probe`
2. `findings/loop1_quant_mechanism.md` — Q4_0 crushes small-magnitude weights (~99% vs ~5–11%
   relative error), AND the explicit list of what this does *not* prove. Rerun:
   `python3 -m bench.quant_demo`
3. `PREREGISTRATION.md` — the draft with the two decisions below flagged for your sign-off.

---

## Decision A — which H1 do you put on the poster?

| Option | Statement | Trade-off |
|---|---|---|
| **H1-basic** | FP16→Q4_0 degradation is disproportionately larger for low-resource languages (a language×quant interaction). | Safer; near-certain to hold; weaker claim. |
| **H1-sharp** (newly motivated) | Degradation is **ordered by resource level: English < Sanskrit < Tibetan**, matching the independently-measured token-sparsity ordering. | Stronger, more falsifiable (the ranking can break even if a gap exists); better science and a better poster line. |

The tokenizer finding is what makes H1-sharp defensible — you'd be predicting the ordering *a
priori* from a measurement you already have, then testing whether quantization reproduces it.

## Decision B — what counts as "regressive" (fix before results)

Absolute vs. relative degradation disagree (Tibetan 40→60% CER and English 5→7.5% WER = +20 vs
+2.5 points, but +50% for both). Draft recommendation: **report both, named distinctly** —
*user-experienced harm* (absolute, primary) and *mechanistic amplification* (relative, stricter).
Alternatives: absolute only, or relative only.

## Decision C — the falsifier (what makes you abandon H1)

- For **H1-basic**: abandon if the absolute interaction-contrast CI includes 0 for both Tibetan
  and Sanskrit. Optional floor-effect guard: if Q4_0 error > 0.80 for English too, the model is
  just broken at 4 bits (no gradient to attribute).
- For **H1-sharp**: additionally abandon if the degradation ordering is violated (e.g. Sanskrit
  degrades more than Tibetan) even when an interaction exists.

---

## Decision D — H2 (Loop 2, the stretch / the "fix")

H2 doesn't need a wording choice the way H1 does, but decide **now** whether it's on the poster
or held for a follow-up post, because it changes what you build after Loop 1.

> **H2.** Part of the quantization tax is a *calibration-data artifact*, not inherent to 4 bits:
> quantization calibrated on Tibetan audio (ggml imatrix; and/or CTranslate2 INT8 calibration
> cache) recovers a significant fraction of the Q4_0 loss at identical bits per weight.

- **Falsifier:** Tibetan-calibrated quant does *not* beat generic quant on Tibetan CER (CI
  includes 0), or it only does so by wrecking English (a trade, not a fix — which is still a
  reportable finding, just a different one).
- **Poster stance to pick:** (a) H1 only, H2 teased as "next"; or (b) H1 + a small H2 panel if
  Loop 2 lands in time. The poster stands on H1 alone either way.
- Detail: `docs/loop2_language_aware.md`. Claim discipline: language-aware quant improves the
  model's fit to the language's *acoustic/orthographic context* — never "understanding."

## State of the build (all committed)

- ✅ Loop 0 harness complete and self-verifying — `python3 -m bench.test_harness` passes
  (normalization, WER/CER, bootstrap CIs, append-only/resumable runner).
- ✅ Docs restructured into the loop program; README rewritten; `docs/` now public.
- ✅ Prereg draft + paper scaffold (`paper/paper.md`, `paper/outline.md`, one-source/many-renders).
- ✅ Two mechanism probes run and written up as findings.
- ✅ Analysis path (`bench/analyze.py`) built + validated on synthetic data: reproduces the
  predicted ordering (bo > sa > en, CIs excluding 0). Poster figures render as pure-Python SVG
  (no matplotlib on host) — renderer is the next code step.
- ⬜ **Blocked on you:** freeze the prereg (Decisions A/B/C for H1, D for H2).
- ⬜ **Blocked on GPU/downloads (next physical step):** `bash models/build.sh` inside a
  jetson-container to build the FP16/Q8_0/Q4_0 ladder, download corpora, then
  `python3 -m bench.runner configs/smoke.json` for the Loop 0 gate.

## Commits this session

```
bdff58e Findings: measure the mechanism before claiming it (Loop 1b + quant demo)
54ec9e9 Loop 1 prereg draft + paper scaffold (one-source, many-renders)
88e885f Part A: restructure docs into loop program, rewrite README, publish docs/
6b5f059 Loop 0: benchmark harness scaffold (schemas, engines, corpora, metrics)
```

Plan of record: `~/.claude/plans/majestic-booping-fog.md`.
