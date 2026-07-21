# Research Program — Loop Index

This project is a **ladder of small research loops**, each with an explicit success gate. A
loop produces a repo module, a Substack post, and (for Loops 1–2) a conference poster panel.
Failing a gate means stop and redesign, not push forward.

The through-line: **how will quantized edge devices — phones, robots — understand rare
languages?** Nobody deploys a full-precision foundation model; they quantize for speed and
memory. This program measures who pays for that compression, and whether the cost falls
hardest on languages already underrepresented in training.

| Loop | Question | Field | Status |
|---|---|---|---|
| [0](loop0_harness.md) | Can I measure anything reproducibly? | infra | 🚧 in progress |
| [1](loop1_quantization_tax.md) | Does quantization tax rare languages more? | ASR | 📋 planned — **poster core** |
| 1b | *Why* — token-level sparsity | ASR | 📋 planned |
| [2](loop2_language_aware.md) | Can language-aware quantization fix it? | ASR | 📋 planned — **stretch** |
| [3–7](roadmap.md) | QLoRA → segmentation → pipeline → LLM → vector index | — | 📋 roadmap (WIP) |

**Sprint scope:** Loops 0–2, targeting the IFDS Research Expo poster (2026-08-10). Loop 1 is
the guaranteed deliverable; Loop 2 and the TensorRT engine arm are nested fallbacks.

- [design_spec.md](design_spec.md) — shared visual/style spec for all report pages (unchanged)
- [roadmap.md](roadmap.md) — Loops 3–7, including the microservice pipeline architecture

The retired `phase1/2/3_*.md` and `architecture.md` planning notes have been folded into these
loop docs and the roadmap; their substantive content is preserved there.
