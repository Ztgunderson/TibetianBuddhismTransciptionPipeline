# From the Lab to the Cushion: Who Pays for Quantization?

*Measuring the disproportionate cost of edge-model compression on a low-resource language,
with Tibetan Buddhist teachings as the case study.*

Zachary Gunderson · Draft · Sections fill in as loops close (see `paper/outline.md`).

---

## Abstract

*(≤150 words — written last.)* Edge devices run quantized models. We ask whether that
compression is fair across languages, and find [HEADLINE — after Loop 1] …

---

## 1. Motivation

*(Draft — from Part A.)* Full-precision foundation models are not what runs at the edge. Before a
speech model reaches a phone or a robot it is quantized — its weights compressed from 16 bits to
8 or 4 — for speed and memory. Benchmarks, however, report full-precision numbers. The accuracy a
real deployment delivers is the *post-compression* one, and whether that number is equally good
across languages is not something published leaderboards tell you.

This matters most for languages that were already underrepresented. [Deer Park context; survey
gap; the fairness question.]

## 2. Background

*(Draft — from Part A.)*

- **Tibetan in AI.** [arXiv 2510.19144 — underserved status; "token-level sparsity and structural
  underrepresentation," asserted but unmeasured. TLUE, FLORES-200 Tibetan track, IARPA Babel
  accessibility barrier.]
- **How block quantization works.** [Per-block scale; high-magnitude weights survive; the
  mechanism by which rare-language subspaces are preferentially crushed.]
- **Why CER, not WER, for Tibetan/Sanskrit.** [Agglutinative morphology; ill-defined word
  boundaries; symmetric normalization — tsheg/shad, danda.]

## 3. Hypotheses & Preregistration

*(Locked in `PREREGISTRATION.md`.)* H1, H1b, H2 as stated in `paper/outline.md`. The interaction
contrast and the falsifier are fixed before any full run.

## 4. Methods

*(Fill after Loop 0 full run.)* Corpora and licenses; the 18-cell grid; per-language
normalization; bootstrap CIs over utterances; Jetson Orin + jetson-containers; both engines.

## 5. Results — The Quantization Tax

*(Fill after Loop 1.)* [Figure 1: WER/CER × quant × language with CIs. Figure 2: RTF + watts.
The interaction contrast and its CI. Absolute and relative framings.]

## 6. Mechanism — Token-Level Sparsity

*(Fill after Loop 1b.)* [Tokens per second of audio by language; byte-fallback evidence.]

## 7. Intervention — Language-Aware Quantization

*(Fill after Loop 2.)* [imatrix generic/en/bo at identical size; INT8 calibration-cache arm;
recovery fraction with CI; honest English cost.]

## 8. Limitations

*(Draft.)* Domain mismatch (studio read speech vs. spontaneous dharma audio); single hardware;
error-source entanglement; opaque provenance of off-the-shelf finetunes (roadmap).

## 9. Implications

*(Fill at end.)* Edge-deployment fairness; concrete recommendation for shipping multilingual ASR;
what this enables for Deer Park's archive.

---

## Ethics & Data Statement

*(Draft — from Part A.)* Volunteer work for a religious community's preservation goals; audio
analyzed but never redistributed; names used with knowledge; corpus licenses (incl. Vāksañcayaḥ
non-commercial) honored.

## Reproducibility

See `KIT.md`. Every figure regenerates from a committed `experiment_id` and its sampled manifests.
