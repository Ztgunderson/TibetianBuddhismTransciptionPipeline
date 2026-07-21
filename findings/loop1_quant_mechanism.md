# Finding — What Q4_0 Actually Does to Weights (and what this does NOT prove)

**Date:** 2026-07-21 · **Status:** mechanism demonstration on synthetic weights
**Reproduce:** `python3 -m bench.quant_demo`
**Claim strength:** the weight-level property is solid and reproducible; the *language* link is a
hypothesis this demo motivates but does NOT establish.

## What Q4_0 does (verified)

Weights are quantized in blocks of 32. Each block gets ONE scale, pinned to its largest-magnitude
weight; every weight is then snapped to a 4-bit multiple of that scale. Measured consequence:

| Weight population | relative reconstruction error under Q4_0 |
|---|---|
| Large-magnitude (top decile) | ~5–11% |
| Small-magnitude (bottom decile) | ~99–100% |
| ratio | **~10–20×** |

Small weights take a far larger *relative* hit because the block's step size is dictated by its
biggest weight, and small weights get flattened toward zero or a single step. This is a genuine,
reproducible property of the algorithm, not a story.

## What this does NOT prove (important — don't overclaim)

The demo shows *small weights are crushed more*. It does **not** show that low-resource-language
features live in small/sparse weights. That second step is the actual H1 hypothesis, and this
synthetic demo cannot establish it. Two honesty checks I had to make:

1. In the demo, the homogeneous case showed a *larger* small/large ratio (19.6×) than the
   heavy-tailed case (9.2×) — because when big weights dominate a block, they also take a bigger
   absolute hit themselves. So "outlier weights crush their neighbors" is real but not cleanly
   monotonic in the way a too-neat telling would suggest. The paper should show the numbers, not a
   tidier cartoon.
2. Whether Tibetan's learned features actually occupy the crushed subspace is an **empirical claim
   about a specific model**, testable only by quantizing the real Whisper weights and measuring
   accuracy per language (= Loop 1). The demo motivates the hypothesis; Loop 1 tests it.

## The honest chain of reasoning (what we can and can't say)

- ✅ Q4_0 gives small-magnitude weights much larger relative error. (demonstrated here)
- ✅ Tibetan costs ~15× more decoder tokens than English. (measured, see
  [loop1b_tokenizer_probe.md](loop1b_tokenizer_probe.md))
- ❓ Tibetan-relevant weights are disproportionately low-magnitude / sparsely activated.
  (plausible, UNTESTED — candidate for a Loop 3+ interpretability probe)
- ❓ Therefore Tibetan accuracy degrades more under quantization. (= H1, tested in Loop 1)

The gap between the ✅ and ❓ lines is precisely why H1 needs an experiment and a preregistered
falsifier rather than a mechanism argument alone. This is the thing to keep straight in the notes:
we can *see* the quantization mechanism and *measure* the tokenizer sparsity today, but the claim
that ties them to a language-fairness gap is exactly what remains to be earned with data.
