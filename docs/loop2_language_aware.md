# Loop 2 — Language-Aware Quantization (stretch)

The first loop with a *fix* attached rather than a problem described — and the cheapest possible
one: no training, no extra parameters, just a different calibration set.

## Hypothesis

> **H2.** Part of the quantization tax is not inherent to 4 bits — it is an artifact of
> calibrating quantization on English-dominant data. Quantization calibrated on Tibetan audio
> recovers a significant fraction of the Q4_0 loss at identical bits per weight.

## The same idea in both engines

- **Engine A (ggml):** importance-matrix (imatrix) quantization — compute an importance matrix
  over a calibration set, weight quantization by it. Build three Q4_0 variants at *identical
  size*: generic/default, English-calibrated, Tibetan-calibrated.
- **Engine B (CTranslate2 / TensorRT INT8):** the INT8 **calibration cache** is literally built
  from representative data. Build one cache from English audio, one from Tibetan, compare. This
  is language-aware quantization in the NVIDIA-native toolchain — the cleanest test of whether
  the effect generalizes beyond one library.

If it shows up in *both* engines, the finding is that generic edge quantization under-serves rare
languages as a matter of *calibration data*, not a quirk of ggml. (Engine B runs only if Loop 1
finished on schedule.)

## Gate

Language-calibrated quant beats generic quant on Tibetan CER, CI excluding zero, without
catastrophic English regression. If it's a tradeoff rather than a free win, that tradeoff *is*
the finding — report the English cost honestly.

## Claim discipline

Language-aware quantization improves how well the model handles **its language's
acoustic-phonetic and orthographic context** — the Tibetan sound patterns and script that
generic English-calibrated quantization crushes. That is a real gain in the model's contextual
fit to the language. It does **not** give semantic/doctrinal *understanding* — Whisper transcribes,
it does not comprehend. Understanding belongs to the LLM tier (roadmap Loop 6, TLUE /
Ti-SafetyBench). Stating it at exactly this resolution matters: too modest undersells the result,
too strong gets the whole piece discounted by an academic or LessWrong reader.
