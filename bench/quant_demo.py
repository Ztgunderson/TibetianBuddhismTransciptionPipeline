"""What does Q4_0 actually DO to a weight? A runnable demonstration of the H1 mechanism.

No GPU, no models — pure numpy on synthetic weights. The point is to *see*, on real numbers, why
block quantization plausibly hurts low-magnitude / sparsely-activated weights more, which is the
mechanistic core of H1.

GGML Q4_0 in one paragraph: weights are cut into blocks of 32. For each block we find the value
with the largest absolute magnitude (`amax`), set a single scale `d = amax / -8`, and store each
weight as a 4-bit integer q in [-8, 7] via `round(w / d)`. Dequantizing gives `w' = q * d`. So the
whole block shares ONE scale, pinned to that block's biggest weight. The reconstruction step size
is `|d|` — every weight is snapped to the nearest multiple of it.

Consequence: the step size is dictated by the block's largest weight. A block dominated by a few
big weights has a coarse step, and the many small weights in it get flattened toward zero or a
single step. If the features a rare language relies on live in low-magnitude weights that share
blocks with high-magnitude (high-resource) weights, quantization preferentially erases them. This
script shows that effect numerically.

Run:  python3 -m bench.quant_demo
"""

from __future__ import annotations

import numpy as np

BLOCK = 32
NBITS = 4
QMIN, QMAX = -8, 7  # 4-bit signed range used by q4_0


def quantize_q4_0(w: np.ndarray) -> np.ndarray:
    """Quantize then dequantize a 1-D array with ggml-style Q4_0 (block=32). Returns w'."""
    n = len(w)
    out = np.empty_like(w, dtype=np.float64)
    for start in range(0, n, BLOCK):
        block = w[start:start + BLOCK]
        amax = np.max(np.abs(block)) if len(block) else 0.0
        if amax == 0:
            out[start:start + BLOCK] = 0.0
            continue
        d = amax / QMIN            # scale pinned to the block's largest-magnitude weight
        q = np.round(block / d).clip(QMIN, QMAX)
        out[start:start + BLOCK] = q * d
    return out


def _report(name: str, w: np.ndarray) -> None:
    wq = quantize_q4_0(w)
    err = np.abs(wq - w)
    mag = np.abs(w)
    # split weights into "large" (top decile by magnitude) vs "small" (bottom decile)
    hi = mag >= np.quantile(mag, 0.9)
    lo = mag <= np.quantile(mag, 0.1)
    # relative error on small vs large weights
    rel_small = np.mean(err[lo] / (mag[lo] + 1e-12))
    rel_large = np.mean(err[hi] / (mag[hi] + 1e-12))
    print(f"\n{name}")
    print(f"  mean abs error            : {err.mean():.5f}")
    print(f"  rel error, LARGE weights  : {rel_large:6.1%}")
    print(f"  rel error, SMALL weights  : {rel_small:6.1%}   <- the crushed ones")
    print(f"  small/large error ratio   : {rel_small / (rel_large + 1e-12):6.1f}x")


def main() -> None:
    rng = np.random.default_rng(0)

    # 1. A homogeneous block: all weights similar magnitude. Quantization is even-handed.
    homo = rng.normal(0, 1, 4096)
    _report("Homogeneous weights (no magnitude outliers):", homo)

    # 2. Heavy-tailed weights: a few large 'high-resource' weights share blocks with many small
    #    'rare-feature' weights. This is the realistic case for a multilingual model.
    heavy = rng.normal(0, 1, 4096)
    spikes = rng.choice(4096, size=128, replace=False)
    heavy[spikes] *= 12.0     # inject high-magnitude weights (frequent-data-reinforced)
    _report("Heavy-tailed weights (big weights share blocks with small ones):", heavy)

    print("\nReading: when large weights dominate a block, they set a coarse step size, and the")
    print("small weights sharing that block take a much larger *relative* hit. If low-resource-")
    print("language features live in those small weights, generic quantization erases them first.")
    print("That is H1's mechanism, on numbers. Loop 1 tests whether it shows up as an accuracy")
    print("gap across languages; Loop 2 tests whether calibrating the scale on Tibetan data helps.")


if __name__ == "__main__":
    main()
