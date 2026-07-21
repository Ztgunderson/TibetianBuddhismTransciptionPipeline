# Roadmap — Loops 3–7 (WIP, out of sprint scope)

These loops are **deliberately under-specified**: they get designed from what Loops 1–2 actually
show. The north star is a pipeline that transcribes Deer Park's audio archive into trustworthy
text, summarizes it, and indexes it into a domain-aware searchable database their team can use to
build classes and maintain the lineage. Each loop keeps its own success gate.

## Loop 3 — QLoRA finetuning (ASR)

> Does adaptation buy back what quantization costs? Test whether a QLoRA-finetuned
> whisper-small at Q4_0 beats base whisper-large at FP16 — i.e. on a fixed edge budget, spend on
> adaptation, not bits.

Train on NICT-Tib1 on the Orin (in a `dustynv` container — known-provenance training data, so no
contamination ambiguity), then quantize. `milanakdj/whisper-small-full-tibetan` is benchmarked as
an off-the-shelf comparison arm, but its training set is opaque, so it is not the primary result.
**Gate:** beats the best Loop 2 baseline.

## Loop 4 — Segmentation benchmarking (from phase2_pipeline)

Real teachings alternate Tibetan, English, and Sanskrit within one recording, so routing precedes
transcription. **Why MMS, not Whisper:** Whisper's language detector is ~99-language forced-choice
with weak Tibetan coverage; MMS ships dedicated LID at 126–4017 languages, far better suited to
detecting Tibetan/Sanskrit turn boundaries.

Pipeline: MMS LID segments by language turn → each segment routed to Whisper with the language
*forced* (no second-pass detection), isolating segmentation accuracy as an independent variable.
Dialect ID via `tiantiaf/voxlect-tibetan-dialect-mms-lid-256` (3-way Amdo/Kham/Ü-Tsang on
mms-lid-256, CC-BY-NC-4.0, 3–15s mono 16kHz chunks) runs *after* LID, as an ensemble.

Language codes: Tibetan `bo` (bod), English `en` (eng), Sanskrit `sa` (san).
**Gate:** boundary accuracy on a hand-checked sample good enough that downstream ASR isn't
dominated by routing errors. **Parked:** audio enhancement (noise/dereverb) — all runs on raw
audio; overlapping speech out of scope.

## Loop 5 — Full audio pipeline (microservice architecture)

Where the microservice design earns its keep. 14 Deer Park clips (8 Geshe Tenzin Sherab,
*Ornament of Clear Realizations*; 6 Geshe Lhundup Jinpa, *Compilation of the Essential
Practices*). No ground truth exists, so the headline measure is **quantization-induced
divergence**: how far Q8_0/Q4_0 output drifts from the same model's FP16 output. Zero annotation
required; the within-recording English/Tibetan contrast controls acoustics for free.

**Gate:** end-to-end run on all 14 clips producing text a Tibetan-literate reviewer calls usable
— the trust threshold for the archive, which needs a human in the loop.

### Service architecture (from the retired architecture.md)

Six stateless services, each doing one job over a small HTTP API, every payload a versioned
schema from `schemas/`, models swappable via env var, results append-only by `experiment_id`,
dashboard reads result files only. Ports: segmentation 8001, asr_edge 8002 (whisper-large, CUDA),
asr_laptop 8003 (whisper-small, CPU-capped to simulate laptop tier), translate 8004 (Hy-MT2),
score 8005, orchestrator 8000. On Jetson: `dustynv/whisper`, `dustynv/ollama`, custom
`dustynv/pytorch` builds for segmentation/score; orchestrator on `python:3.12-slim`. The
orchestrator wraps runs in `tegrastats` and locks clocks with `jetson_clocks`. Data flow:
`raw_audio → segmentation → asr → translate (Tibetan) → score → results → dashboard`.

## Loop 6 — Meaning preservation + LLM benchmarking (from phase3_meaning)

> Does the doctrinal signal survive transcription + translation — not just look similar?

Five methods run **side by side, not pre-combined**, treating their divergence as the finding
(per arXiv 2603.17303, which shows sentence-similarity metrics including COMET are unreliable
proxies for culturally-loaded meaning): (1) cosine similarity (fast triage only), (2) key-point
extraction & verification (QAGS-style — the primary signal), (3) COMET-KIWI (reference-free QE),
(4) roundtrip consistency, (5) NLI/entailment (catches dropped qualifiers like "conventionally,"
"ultimately," "according to Prasangika" that are load-bearing in Buddhist texts). Source texts:
Abhisamayalamkara commentary, Lam Rim. Reference: the human interpreter's production translation.

**LLM benchmarking** uses TLUE (Ti-MMLU + Ti-SafetyBench) and FLORES-200's Tibetan track.
Ti-SafetyBench makes safety alignment in a low-resource language directly measurable — the most
AI-safety-track-relevant work in the program.

**Caveat:** none of the five methods separates ASR error, segmentation error, interpreter
paraphrase, and genuine mistranslation. Read results as "pipeline output vs. interpreter
reference," not "machine vs. ground-truth meaning."

## Loop 7 — Summaries + domain-aware vector index (LLM)

Summarize the transcribed archive and index it into a domain-aware searchable vector database.
The north star: Deer Park's team searching decades of teachings by concept, not filename.
