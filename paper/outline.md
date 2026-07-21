# Paper — Canonical Outline & Format

**One source, many renders.** `paper/paper.md` is the canonical long-form. Everything else is a
projection of it, so a result is written once:

| Render | Audience | How it's derived |
|---|---|---|
| `paper/paper.md` | canonical | the master document; written incrementally as loops close |
| Substack post per loop | general + practitioners | one loop's sections, lightly narrativized, figures inline |
| LessWrong crosspost | AI safety/alignment | same as Substack + a sharper "why this is a fairness/interp result" framing |
| IFDS poster | expo | condensed claims + the Loop 1 figures from `reports/` |

Figures are authored once under `reports/` (HTML/PNG, styled per `docs/design_spec.md`) and
embedded by every render. No figure is redrawn per format.

---

## Hypotheses (locked before results)

- **Loop 0 — no hypothesis.** Infrastructure. Success = the harness self-test passes and the
  smoke gate produces valid `TranscriptRow`s with plausible English WER on GPU.
- **H1 (Loop 1).** FP16→Q4_0 degradation is disproportionately larger for low-resource languages
  than English — a language × quantization *interaction*. Primary statistic and falsifier fixed
  in `PREREGISTRATION.md`.
- **H1b (Loop 1b).** Tibetan requires more decoder tokens per second of audio than English under
  Whisper's BPE (byte-fallback on Tibetan script), giving a mechanistic account of part of H1.
- **H2 (Loop 2).** Language-aware (Tibetan-calibrated) quantization recovers a significant
  fraction of the Q4_0 loss at identical bits per weight — the tax is partly a calibration-data
  artifact, not inherent to 4 bits.

---

## Section structure (paper.md)

| § | Section | Filled after |
|---|---|---|
| Abstract | 150 words: the fairness framing + headline number | end |
| 1 | **Motivation** — nobody deploys FP16; Deer Park's preservation need; the survey gap | Part A ✅ |
| 2 | **Background** — Tibetan in AI (2510.19144); how block quantization works; why CER | Part A ✅ |
| 3 | **Hypotheses & preregistration** — H1/H1b/H2, the interaction contrast, the falsifier | prereg |
| 4 | **Methods** — corpora + licenses, the grid, normalization, bootstrap CIs, Orin/containers, both engines | Loop 0 |
| 5 | **Results: the quantization tax** — the interaction, per language, with CIs; RTF + watts | Loop 1 |
| 6 | **Mechanism: token-level sparsity** — tokens/sec per language; layer-wise if time | Loop 1b |
| 7 | **Intervention: language-aware quantization** — imatrix + INT8 calibration, both engines | Loop 2 |
| 8 | **Limitations** — domain mismatch (read vs. spontaneous); single hardware; opaque finetune provenance; error-source entanglement | Loop 1 |
| 9 | **Implications** — edge deployment fairness; what it means for Deer Park; what to do about it | end |
| — | **Ethics & data statement** — consent, non-redistribution, corpus licenses | Part A ✅ |
| — | **Reproducibility** — pointer to KIT.md, experiment_ids, committed manifests | Loop 1 |

## Writing rules

- Every claim ties to a figure or a CI. No adjective the data doesn't earn.
- Claim discipline (Loop 2): "improves the model's fit to the language's acoustic/orthographic
  **context**" — never "understanding." Understanding is the LLM tier (roadmap Loop 6).
- Report nulls as findings. A failed H1 is a publishable result, framed as such.
- Absolute *and* relative degradation both reported, named per the prereg — never one silently
  standing for the other.
