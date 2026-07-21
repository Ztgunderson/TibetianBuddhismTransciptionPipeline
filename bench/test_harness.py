"""Self-test for the Loop 0 harness — runs with no corpora, no models, no GPU.

Proves the pieces that must be correct before any hypothesis is tested:
  - normalization does not manufacture cross-language error differences
  - WER/CER and RTF compute correctly
  - bootstrap CIs are reproducible and behave (identical sets straddle zero)
  - the runner writes valid TranscriptRows, is append-only, and resumes without duplication

Uses a stub engine and synthetic silent wavs so it runs anywhere:
    python3 -m bench.test_harness
Exits nonzero on any failure.
"""

from __future__ import annotations

import json
import sys
import tempfile
import wave
from pathlib import Path

from bench.metrics import error_rate, rtf, bootstrap_error_rate, bootstrap_difference
from bench.normalize import normalize


def _make_silence(path: Path, seconds: float = 1.0, rate: int = 16000) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"\x00\x00" * int(rate * seconds))


def check(name: str, cond: bool) -> None:
    print(f"  [{'ok' if cond else 'FAIL'}] {name}")
    if not cond:
        check.failed = True  # type: ignore[attr-defined]


check.failed = False  # type: ignore[attr-defined]


def test_metrics() -> None:
    print("metrics + normalization")
    check("english WER = 0.25 for 1/4 substitution", error_rate("the cat sat down", "the dog sat down", "en") == 0.25)
    check("case/punct folding -> 0", error_rate("The Cat, sat!", "the cat sat", "en") == 0.0)
    check("tibetan tsheg vs space -> 0 (load-bearing)", error_rate("བཀྲ་ཤིས་", "བཀྲ ཤིས", "bo") == 0.0)
    check("tibetan shad stripped symmetrically", error_rate("བཀྲ་ཤིས།", "བཀྲ་ཤིས", "bo") == 0.0)
    check("sanskrit danda stripped", error_rate("धर्म ।", "धर्म", "sa") == 0.0)
    check("rtf = 0.25", rtf(0.5, 2.0) == 0.25)
    check("normalize is idempotent", normalize(normalize("The  Cat.", "en"), "en") == normalize("The  Cat.", "en"))


def test_bootstrap() -> None:
    print("bootstrap CIs")
    pairs = [("the cat sat", "the cat sat", "en")] * 10 + [("a b c", "a b d", "en")] * 10
    e1 = bootstrap_error_rate(pairs, n_resamples=500, seed=1)
    e2 = bootstrap_error_rate(pairs, n_resamples=500, seed=1)
    check("reproducible with fixed seed", e1.point == e2.point and e1.lo == e2.lo)
    check("point estimate in [lo, hi]", e1.lo <= e1.point <= e1.hi)
    d = bootstrap_difference(pairs, pairs, n_resamples=500, seed=1)
    check("identical sets: difference CI straddles zero", d.lo <= 0.0 <= d.hi)


def test_runner() -> None:
    print("runner: schema, append-only, resume")
    from corpora.base import Utterance
    from engines.base import Engine, Transcription
    import bench.runner as R

    class StubEngine(Engine):
        name = "stub"

        def __init__(self, model_path: str, **_):
            self.model_path = model_path

        def transcribe(self, audio_path: str, lang: str) -> Transcription:
            return Transcription(text="the cat sat", audio_seconds=1.0, compute_seconds=0.3)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        wav = tmp / "u1.wav"
        _make_silence(wav)
        manifest = [
            Utterance("u1", str(wav), "the cat sat", "en", "stub"),
            Utterance("u2", str(wav), "the dog sat", "en", "stub"),
        ]
        results = tmp / "transcripts.jsonl"
        cell = R.Cell(engine="stub", model="m", quant="fp16", model_path="x")

        # patch the registry to return our stub
        orig = R.get_engine
        R.get_engine = lambda name, **cfg: StubEngine(**cfg)
        try:
            n1 = R.run_cell(cell, manifest, "exp", results, R._done_keys(results))
            rows = [json.loads(l) for l in results.read_text().splitlines() if l.strip()]
            check("wrote one row per utterance", n1 == 2 and len(rows) == 2)
            check("rows validate as TranscriptRow", all("schema_version" in r for r in rows))
            check("english row has wer, not cer", rows[0]["wer"] is not None and rows[0]["cer"] is None)
            # resume: second run should add nothing
            n2 = R.run_cell(cell, manifest, "exp", results, R._done_keys(results))
            rows2 = [l for l in results.read_text().splitlines() if l.strip()]
            check("resume writes no duplicates", n2 == 0 and len(rows2) == 2)
        finally:
            R.get_engine = orig


def main() -> int:
    test_metrics()
    test_bootstrap()
    test_runner()
    print()
    if check.failed:  # type: ignore[attr-defined]
        print("HARNESS SELF-TEST FAILED")
        return 1
    print("harness self-test passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
