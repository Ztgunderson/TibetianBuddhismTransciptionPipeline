"""The benchmark sweep: model x quant x engine x corpus -> TranscriptRow[] on disk.

Design constraints (from docs/architecture.md, carried into Loop 0):
  - Append-only, keyed by experiment_id. A 6-hour run that dies at hour 5 keeps hours 0-5.
  - Resumable: re-running skips cells already present in the results file.
  - Config-driven: scaling the smoke test to the full grid is a config change, not a code edit.
  - Every run is wrapped in tegrastats so hardware telemetry is attached automatically.

Results are written as JSONL (one TranscriptRow per line) to
  data/results/{experiment_id}/transcripts.jsonl
so partial results are readable while a run is still going.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

from corpora import load_corpus, sample_manifest
from corpora.base import write_manifest
from engines import get_engine
from bench.metrics import error_rate, rtf
from bench.telemetry import tegrastats, lock_clocks
from schemas import TranscriptRow

RESULTS_ROOT = Path("data/results")


@dataclass
class Cell:
    """One configuration in the grid — everything needed to instantiate an engine."""

    engine: str          # 'ggml' | 'ct2'
    model: str           # 'whisper-large-v3' | 'whisper-small'
    quant: str           # 'fp16' | 'q8_0' | 'q4_0' | 'int8'
    model_path: str      # path to the weights for this (engine, model, quant)
    calibration: str | None = None   # Loop 2: 'generic' | 'en' | 'bo'
    engine_cfg: dict = field(default_factory=dict)

    def key(self) -> str:
        return f"{self.engine}:{self.model}:{self.quant}:{self.calibration or 'none'}"


def _done_keys(results_path: Path) -> set[str]:
    """(cell_key, utterance_id) pairs already written, for resume."""
    done: set[str] = set()
    if results_path.exists():
        with results_path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    r = json.loads(line)
                    cell_key = f"{r['engine']}:{r['model']}:{r['quant']}:{r.get('calibration') or 'none'}"
                    done.add(f"{cell_key}|{r['utterance_id']}")
    return done


def run_cell(cell: Cell, manifest, experiment_id: str, results_path: Path, done: set[str]) -> int:
    """Transcribe every utterance in `manifest` under `cell`, appending rows. Returns count written."""
    engine = get_engine(cell.engine, model_path=cell.model_path, **cell.engine_cfg)
    if not engine.health():
        print(f"  ! engine unhealthy for {cell.key()} — skipping", file=sys.stderr)
        return 0
    written = 0
    with results_path.open("a", encoding="utf-8") as out:
        for u in manifest:
            if f"{cell.key()}|{u.utterance_id}" in done:
                continue
            t = engine.transcribe(u.audio_path, u.lang)
            row = TranscriptRow(
                utterance_id=u.utterance_id, corpus=u.corpus, lang=u.lang,
                dialect=u.dialect, speaker_id=u.speaker_id,
                engine=cell.engine, model=cell.model, quant=cell.quant, calibration=cell.calibration,
                hypothesis=t.text, reference=u.reference,
                wer=error_rate(t.text, u.reference, u.lang) if u.lang == "en" else None,
                cer=error_rate(t.text, u.reference, u.lang) if u.lang != "en" else None,
                rtf=rtf(t.compute_seconds, t.audio_seconds),
                audio_seconds=t.audio_seconds, compute_seconds=t.compute_seconds,
                experiment_id=experiment_id,
            )
            out.write(row.model_dump_json() + "\n")
            out.flush()  # append-only durability: survive a crash mid-cell
            written += 1
    engine.close()
    return written


def run_sweep(config: dict) -> Path:
    """Execute a full grid described by `config`. See configs/ for the shape."""
    experiment_id = config["experiment_id"]
    exp_dir = RESULTS_ROOT / experiment_id
    exp_dir.mkdir(parents=True, exist_ok=True)
    results_path = exp_dir / "transcripts.jsonl"

    if not lock_clocks():
        print("! jetson_clocks not applied (need sudo / not on Jetson) — RTF may vary", file=sys.stderr)

    # Build the sampled manifest once per corpus and persist it for exact reproducibility.
    manifests = {}
    for corpus, spec in config["corpora"].items():
        full = load_corpus(corpus)
        sampled = sample_manifest(
            full, minutes=spec.get("minutes"), max_utterances=spec.get("max_utterances"),
            seed=config.get("seed", 0), stratify_by=spec.get("stratify_by"),
        )
        write_manifest(sampled, exp_dir / f"manifest_{corpus}.jsonl")
        manifests[corpus] = sampled
        print(f"  corpus {corpus}: {len(sampled)} utterances sampled")

    cells = [Cell(**c) for c in config["cells"]]
    done = _done_keys(results_path)

    with tegrastats(exp_dir / "tegrastats.log"):
        for cell in cells:
            for corpus in config["cell_corpora"].get(cell.key(), config["corpora"].keys()):
                n = run_cell(cell, manifests[corpus], experiment_id, results_path, done)
                print(f"  {cell.key()} x {corpus}: +{n} rows")
    print(f"done -> {results_path}")
    return results_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an ASR benchmark sweep.")
    ap.add_argument("config", help="Path to a JSON sweep config (see configs/).")
    args = ap.parse_args()
    run_sweep(json.loads(Path(args.config).read_text(encoding="utf-8")))


if __name__ == "__main__":
    main()
