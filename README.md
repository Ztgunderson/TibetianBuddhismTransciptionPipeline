# From the Lab to the Cushion

**Do quantized edge models understand rare languages — or does compression tax the languages
that were already underrepresented?**

An open, reproducible study of speech-to-text for **Tibetan Buddhist teachings** (Tibetan,
Sanskrit, English), run entirely on a Jetson AGX Orin. It measures what actually happens to a
low-resource language when you do the thing everyone does before deploying a model to a phone or
a robot: **quantize it.**

> Poster: IFDS Research Expo, August 2026. Findings: a Substack series (links added as they
> publish). This is a volunteer project supporting [Deer Park Buddhist
> Center](https://www.deerparkcenter.org)'s effort to preserve their recorded teachings.

---

## Why this exists

Nobody runs a full-precision foundation model at inference time. Phones and edge devices run
**quantized** weights — smaller, faster, cheaper. So the accuracy that matters for a real
preservation project is the number *after* compression, not the published benchmark.

The survey *[Tibetan Language and AI](https://arxiv.org/html/2510.19144v1)* documents Tibetan as
one of the most underserved languages in AI, attributing the failure to *"token-level sparsity
and structural underrepresentation."* It asserts that mechanism but never measures it. **This
project measures it** — and then asks whether it can be fixed cheaply enough to run on a device a
community center could afford.

If generic compression falls hardest on exactly the languages already underrepresented in
training, then routine edge deployment silently widens the gap. That is a fairness question, it
is measurable, and it has not been measured for Tibetan.

---

## Research arc

A ladder of small loops, each with an explicit success gate and a write-up. Full detail in
[`docs/`](docs/README.md).

| Loop | Question | Status |
|---|---|---|
| **0** | Can I measure anything reproducibly? | 🚧 in progress |
| **1** | Does quantization tax rare languages more? (poster core) | 📋 planned |
| **1b** | *Why* — token-level sparsity, measured | 📋 planned |
| **2** | Can *language-aware* quantization fix it? (stretch) | 📋 planned |
| 3–7 | QLoRA → segmentation → full pipeline → LLM → searchable archive | 📋 [roadmap](docs/roadmap.md) |

**H1 (Loop 1):** degradation from FP16 → 4-bit is disproportionately larger for Tibetan and
Sanskrit than for English — a language × quantization *interaction*, not just a main effect.

**H2 (Loop 2):** part of that tax isn't inherent to 4 bits — it's an artifact of calibrating
quantization on English-dominant data. Calibrating on Tibetan audio recovers much of the loss at
identical model size.

Hypotheses are **preregistered** (`PREREGISTRATION.md`, committed before any full run) so results
are confirmatory, not post-hoc.

---

## Method at a glance

- **Models:** whisper-large-v3 and whisper-small, each at **FP16 / Q8_0 / Q4_0**.
- **Engines:** whisper.cpp (ggml, CUDA) primary; faster-whisper / CTranslate2 (INT8) as a second
  arm for engine-independence.
- **Corpora (all openly licensed):** FLEURS (English), Vāksañcayaḥ (Sanskrit, IIT Bombay),
  NICT-Tib1 (Tibetan, OpenSLR 158), TIBMD@MUC (Tibetan dialects, OpenSLR 124).
- **Metrics:** WER (English), CER (Tibetan/Sanskrit), RTF, plus measured **watts and memory per
  quant level** from `tegrastats` — the edge-specific data. Bootstrap confidence intervals over
  utterances throughout.
- **Platform:** everything GPU-accelerated on a **Jetson AGX Orin** inside NVIDIA
  `jetson-containers`. (The host Python torch is CPU-only, so all model work is containerized.)

An honest note on Tibetan data: the open corpora that exist are *studio read speech in Lhasa
dialect* — a real domain mismatch with reverberant, extemporaneous dharma teaching. That gap
between benchmark and reality is treated as a finding, not hidden.

---

## Repository layout

```
schemas/     versioned data contracts (pydantic) — the single source of truth
corpora/     per-corpus loaders → uniform manifest + deterministic sampling
engines/     ggml (whisper.cpp) and ct2 (CTranslate2) behind one transcribe() interface
bench/       normalization, WER/CER, bootstrap CIs, tegrastats telemetry, the sweep runner
models/      whisper.cpp+CUDA build script → the FP16/Q8_0/Q4_0 ladder
configs/     sweep configs (smoke test → full grid is a config change, not a code change)
docs/        the loop ladder, per-loop methodology, design spec, roadmap
paper/       the write-up, drafted incrementally as loops complete
reports/     per-loop HTML report pages (poster/Substack-ready figures)
data/        corpora + results (gitignored; audio is never committed)
```

## Reproduce

```bash
# 1. Harness self-test — no corpora, models, or GPU required. Run this first.
python3 -m bench.test_harness

# 2. Build the model ladder (inside a jetson-container; see models/build.sh header).
bash models/build.sh

# 3. Smoke test the full sweep on 10 utterances/corpus.
python3 -m bench.runner configs/smoke.json
```

A full reproduction guide (`KIT.md`) — written for someone at Deer Park, not just the author —
lands with Loop 1.

---

## Ethics & attribution

This is volunteer work supporting a religious community's own preservation goals. Recordings are
analyzed but **never redistributed** — no audio is committed to this repo. Teacher and center
names appear with their knowledge. Corpus licenses (including Vāksañcayaḥ's non-commercial terms)
are honored throughout.

## References

- *Tibetan Language and AI: A Comprehensive Survey* — arXiv 2510.19144
- Meta AI, *Scaling Speech Technology to 1,000+ Languages* — arXiv 2305.13516 (MMS)
- *From Words to Worlds: Benchmarking Cross-Cultural Understanding in MT* — arXiv 2603.17303
- NICT-Tib1 (OpenSLR 158, CC BY 4.0) · TIBMD@MUC (OpenSLR 124, CC BY-SA 4.0) · Vāksañcayaḥ
  (IIT Bombay, CC BY-NC 4.0) · FLEURS (CC BY 4.0)

## License

Code under this repository's `LICENSE`. Datasets and model weights retain their upstream
licenses and are not redistributed here.
