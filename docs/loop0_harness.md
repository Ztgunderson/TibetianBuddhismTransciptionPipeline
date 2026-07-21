# Loop 0 — Reproducible Benchmark Harness

**No hypothesis.** The goal is a measurement apparatus trustworthy enough that its numbers can
carry a fairness claim, and reproducible enough that someone at Deer Park can rerun it.

## What it builds

| Module | Job |
|---|---|
| `schemas/` | Versioned pydantic contracts (`TranscriptRow`, `Segment`, `ScoreRow`) — imported everywhere, never re-invented |
| `corpora/` | One loader per corpus → uniform `Utterance` manifest; deterministic seeded sampling |
| `engines/` | `ggml` (whisper.cpp+CUDA) and `ct2` (CTranslate2/INT8) behind one `transcribe()` interface |
| `bench/normalize.py` | Per-language text normalization — the load-bearing correctness piece (below) |
| `bench/metrics.py` | WER (English), CER (Tibetan/Sanskrit), RTF, bootstrap CIs over utterances |
| `bench/telemetry.py` | `tegrastats` wrapper — power/memory per quant level; GPU-execution sanity check |
| `bench/runner.py` | Append-only, resumable model × quant × engine × corpus sweep |
| `models/build.sh` | Builds the FP16/Q8_0/Q4_0 ggml ladder in a jetson-container |

## Why normalization is load-bearing

The headline result is a *difference in error rates between languages*. If Tibetan text is
normalized more or less aggressively than English, that difference is manufactured by the
normalizer, not the model. So every decision in `bench/normalize.py` is explicit and symmetric
across languages — punctuation and spacing only, never linguistic content:

- **Tibetan tsheg** (་, U+0F0B) is the syllable delimiter (the "space" of Tibetan) — kept as a
  content boundary; spacing *style* (space vs. tsheg) is folded so it is never scored as error.
- **Tibetan shad** (།, U+0F0D) is a sentence terminator — stripped, exactly as `.` is for English.
- **Sanskrit danda** (।/॥) — stripped symmetrically with English `.`.
- NFC (not NFKC) composition — some NFKC mappings alter Tibetan stacks.

## Platform

Runs GPU-accelerated **on the Jetson Orin inside `dustynv`/`jetson-containers` images**. This is
forced: the host Python torch is CPU-only (`torch.cuda.is_available() == False`), so any
bare-metal run silently ignores the GPU. `bench/telemetry.py` doubles as the check — a run with
all-zero GPU rows in `tegrastats` fell back to CPU.

## Corpora

| Language | Corpus | Size | License | Source |
|---|---|---|---|---|
| English | FLEURS | ~10h test | CC BY 4.0 | `google/fleurs` (HF) |
| Sanskrit | Vāksañcayaḥ | 78h | CC BY-**NC** 4.0 | IIT Bombay CISTS |
| Tibetan (Lhasa) | NICT-Tib1 | 33.5h read | CC BY 4.0 | OpenSLR 158 |
| Tibetan dialects | TIBMD@MUC | Ü-Tsang 52.5h / Amdo 25.9h / Kham 5.9h | CC BY-SA 4.0 | OpenSLR 124 |

The non-commercial license on Vāksañcayaḥ is noted in the paper. Audio is never committed.

## Gate

10-utterance smoke test (`configs/smoke.json`) across all six Engine-A variants produces valid
`TranscriptRow`s with plausible English WER and a non-empty `tegrastats` log. Scaling to the
full Loop 1 grid must be a config change, not a code change.

The harness self-test (`python3 -m bench.test_harness`) proves normalization, metrics, bootstrap,
and the runner's append-only/resume behavior with no corpora, models, or GPU — run it first.
