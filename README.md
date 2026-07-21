# From the Lab to the Cushion

**Benchmarking Speech-to-Text Pipelines for Tibetan Buddhist Teachings**

A research project examining ASR accuracy, multilingual segmentation, and meaning preservation
across English, Tibetan, and Sanskrit audio — from controlled benchmark conditions to real-world
dharma recordings.

Presented as a poster at conference, August 2026.
Published findings: Substack series (links forthcoming).

---

## Research Question

Can an edge-deployable transcription pipeline (Jetson GPU + CPU fallback) accurately capture the
doctrinal content of mixed-language Tibetan Buddhist teachings — and how do we even measure
"accurately" when the signal is philosophical rather than purely lexical?

---

## Project Structure

```
.
├── README.md                  ← this file
├── docs/
│   ├── design_spec.md         ← shared visual/style spec for all report pages
│   ├── phase1_benchmarking.md ← Phase 1: known-dataset WER/CER/RTF benchmarks
│   ├── phase2_pipeline.md     ← Phase 2: MMS segmentation + pipeline architecture
│   └── phase3_meaning.md      ← Phase 3: meaning preservation evaluation methods
├── data/                      ← datasets (not committed; see phase docs for sources)
└── src/                       ← pipeline code (to be built after planning phase)
```

---

## Model Roster

| Model | Role | Tier |
|---|---|---|
| whisper-large (large-v3) | Primary ASR | GPU / Jetson (CUDA) |
| whisper-small | Comparison ASR | CPU / laptop |
| MMS (facebook/mms-lid-126 … 4017-lang) | Language ID + segmentation only | Edge-deployable |

MMS is used **exclusively as the segmentation and language-ID stage** — not as a competing
transcription model. Its 4,017-language LID capability makes it a better fit for detecting
Tibetan/Sanskrit turns than Whisper's ~99-language forced-choice detector.

---

## Three Phases

| Phase | Title | Question |
|---|---|---|
| 1 | Benchmarking on Known Datasets | How accurate are the models under controlled conditions? |
| 2 | Pipeline + MMS Segmentation | How well does MMS route mixed-language audio? |
| 3 | Meaning Preservation | Does the doctrinal signal survive transcription + translation? |

Each phase produces one report page (paper-style HTML/PDF), styled per `docs/design_spec.md`.

---

## Quantization Levels

Three levels tested across both ASR models:

- **FP16** — full-precision baseline
- **Q8_0** — 8-bit quantization
- **Q4_0** — 4-bit quantization (maximum compression, edge target)

---

## References

- Meta AI, *"Scaling Speech Technology to 1,000+ Languages"*, arXiv 2305.13516
- OpenVINO team, *"MMS: Scaling Speech Technology to 1000+ languages with OpenVINO™"* (notebook)
- *"From Words to Worlds"*, arXiv 2603.17303 (2026) — on meaning preservation vs. similarity metrics
- Vāksañcayaḥ Sanskrit ASR corpus, IIT Bombay, 78 hrs, CC BY-NC 4.0
- FLEURS benchmark (English baseline)
