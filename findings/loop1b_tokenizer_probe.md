# Finding — Token-Level Sparsity Is Real, and It Predicts the Ordering

**Date:** 2026-07-21 · **Status:** preliminary (illustrative samples; corpus rerun pending)
**Reproduce:** `python3 -m bench.tokenizer_probe`
**Claim strength:** strong on direction/ordering; exact magnitudes will move on the real corpora.

## What was measured

How many Whisper decoder tokens each language costs *per character of written content*, using
Whisper's own multilingual BPE tokenizer. CPU-only, no model inference — just tokenization.

| Language | chars/token | tokens/char | bytes/token | step multiplier vs. English |
|---|---|---|---|---|
| English | 5.33 | 0.19 | 5.33 | 1.0× |
| Sanskrit | 0.86 | 1.16 | 2.46 | **6.0×** |
| Tibetan | 0.33 | 3.00 | 1.00 | **15.6×** |

(Illustrative 3-sentence samples per language. The paper number comes from
`--manifest` over the full corpora.)

## What it means

1. **`bytes/token ≈ 1.00` for Tibetan is the smoking gun.** It means the tokenizer found
   essentially *no* useful merges for Tibetan script and fell back to encoding it one UTF-8 byte
   at a time. English, by contrast, merges ~5 bytes into each token. This is exactly the
   "token-level sparsity" the survey (arXiv 2510.19144) names but never quantifies — here it is
   as a number.

2. **The same meaning costs Tibetan ~15× more decoder steps than English.** Whisper decodes
   autoregressively, one token at a time, each conditioned on the last. More steps = more places
   for an error to be introduced and then compounded by the errors before it. A model that must
   emit 15× more tokens for the same content has 15× more opportunities to derail — *before*
   quantization even enters the picture.

3. **The ordering English < Sanskrit < Tibetan matches H1's predicted degradation ordering.**
   This is the important part. H1 predicts quantization damage grows with corpus scarcity
   (English < Sanskrit < Tibetan). Tokenizer sparsity — measured independently, with no
   quantization involved — *already* falls in that exact order. So token sparsity is a plausible
   upstream cause of the H1 gradient, not just a correlate.

## Why this matters for the prereg (the decision I was stuck on)

I was asking you to pick H1's falsifier in the abstract. This finding makes the choice concrete:

- It gives H1 a **mechanistic prior**: we now expect the degradation ordering to track the
  tokens/char ordering. That is a sharper, more falsifiable prediction than "low-resource
  languages degrade more."
- It suggests a **confound to control**: some of Tibetan's ASR difficulty is the *tokenizer*, not
  the acoustic model's weights. Quantization degrades weights. If Tibetan degrades more under
  quantization, is that the weights being crushed (H1's mechanism) or the long token sequence
  amplifying whatever damage occurs? Loop 1 should report degradation **per token emitted**, not
  only per utterance, to separate these. (New analysis column — cheap to add.)

## Open / to verify on real corpora

- Rerun over full NICT-Tib1 / Vāksañcayaḥ / FLEURS references and report the *distribution*
  (median + IQR) of tokens/char, not a 3-sentence point estimate.
- Check whether the multiplier is stable across utterance lengths (byte-fallback should make it
  roughly length-independent — worth confirming).
- Confirm the tokenizer version matches the one the ggml models ship with, so the probe describes
  the same decoder the benchmark uses.
