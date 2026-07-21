"""Concrete corpus loaders and the name->loader registry.

Loaders read each corpus in its native on-disk layout under DATA_ROOT and normalize audio to
16 kHz mono wav (whisper.cpp's required input) in a cache dir the first time an utterance is
requested. Download instructions live in docs/loop0_harness.md — nothing here fetches data.

Expected layout under $ASR_DATA_ROOT (default ./data):
  fleurs/        HuggingFace 'google/fleurs' en_us test split, or {audio/*.wav + metadata.tsv}
  vaksancayah/   {wavs/*.wav, transcripts.tsv}  (utt_id \t text)
  nict_tib1/     Kaldi-style: {wav.scp, text}   (utt_id path ; utt_id text)
  tibmd/         {<dialect>/wav.scp, <dialect>/text} for dialect in u-tsang|kham|amdo
"""

from __future__ import annotations

import os
import subprocess
import wave
from pathlib import Path

from corpora.base import Manifest, Utterance

DATA_ROOT = Path(os.environ.get("ASR_DATA_ROOT", "data"))
CACHE_ROOT = Path(os.environ.get("ASR_WAV_CACHE", DATA_ROOT / "_wav16k"))


# --------------------------------------------------------------------------- audio helpers
def to_wav16k_mono(src: str | Path, cache_key: str) -> tuple[str, float]:
    """Resample any audio to 16 kHz mono wav (cached). Returns (wav_path, duration_seconds).

    Uses ffmpeg, present in the dustynv audio containers. Idempotent: a cached wav is reused.
    """
    src = Path(src)
    out = CACHE_ROOT / f"{cache_key}.wav"
    out.parent.mkdir(parents=True, exist_ok=True)
    if not out.exists():
        subprocess.run(
            ["ffmpeg", "-nostdin", "-y", "-i", str(src),
             "-ar", "16000", "-ac", "1", "-f", "wav", str(out)],
            capture_output=True, check=True,
        )
    with wave.open(str(out), "rb") as w:
        seconds = w.getnframes() / float(w.getframerate())
    return str(out), seconds


def _read_kaldi(scp: Path, text: Path) -> dict[str, tuple[str, str]]:
    """Parse Kaldi wav.scp + text into {utt_id: (audio_path, reference)}."""
    audio: dict[str, str] = {}
    for line in scp.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        utt, rest = line.split(maxsplit=1)
        # wav.scp may hold a path or a piped command; take the first token that looks like a file
        tok = next((t for t in rest.split() if t.endswith((".wav", ".flac", ".mp3", ".opus"))), rest.strip())
        audio[utt] = tok
    refs: dict[str, str] = {}
    for line in text.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        utt, *words = line.split()
        refs[utt] = " ".join(words)
    return {utt: (audio[utt], refs[utt]) for utt in audio if utt in refs}


def _read_tsv(tsv: Path) -> dict[str, str]:
    """utt_id<TAB>text -> {utt_id: text}."""
    out: dict[str, str] = {}
    for line in tsv.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        utt, text = line.split("\t", 1)
        out[utt] = text.strip()
    return out


# --------------------------------------------------------------------------- loaders
def load_fleurs(resample: bool = True) -> Manifest:
    root = DATA_ROOT / "fleurs"
    meta = _read_tsv(root / "metadata.tsv")
    out: Manifest = []
    for utt, text in meta.items():
        src = root / "audio" / f"{utt}.wav"
        path, secs = (to_wav16k_mono(src, f"fleurs/{utt}") if resample else (str(src), None))
        out.append(Utterance(utt, path, text, "en", "fleurs", audio_seconds=secs))
    return out


def load_vaksancayah(resample: bool = True) -> Manifest:
    root = DATA_ROOT / "vaksancayah"
    meta = _read_tsv(root / "transcripts.tsv")
    out: Manifest = []
    for utt, text in meta.items():
        src = root / "wavs" / f"{utt}.wav"
        path, secs = (to_wav16k_mono(src, f"vaksancayah/{utt}") if resample else (str(src), None))
        out.append(Utterance(utt, path, text, "sa", "vaksancayah", audio_seconds=secs))
    return out


def load_nict_tib1(resample: bool = True) -> Manifest:
    root = DATA_ROOT / "nict_tib1"
    pairs = _read_kaldi(root / "wav.scp", root / "text")
    out: Manifest = []
    for utt, (audio, ref) in pairs.items():
        src = audio if Path(audio).is_absolute() else root / audio
        path, secs = (to_wav16k_mono(src, f"nict_tib1/{utt}") if resample else (str(src), None))
        # NICT-Tib1 is single-dialect Lhasa (Ü-Tsang branch); label it for the dialect axis.
        out.append(Utterance(utt, path, ref, "bo", "nict_tib1", dialect="u-tsang", audio_seconds=secs))
    return out


def load_tibmd(resample: bool = True) -> Manifest:
    root = DATA_ROOT / "tibmd"
    out: Manifest = []
    for dialect in ("u-tsang", "kham", "amdo"):
        d = root / dialect
        if not (d / "wav.scp").exists():
            continue
        for utt, (audio, ref) in _read_kaldi(d / "wav.scp", d / "text").items():
            src = audio if Path(audio).is_absolute() else d / audio
            path, secs = (to_wav16k_mono(src, f"tibmd/{dialect}/{utt}") if resample else (str(src), None))
            out.append(Utterance(utt, path, ref, "bo", "tibmd", dialect=dialect, audio_seconds=secs))
    return out


CORPORA = {
    "fleurs": load_fleurs,
    "vaksancayah": load_vaksancayah,
    "nict_tib1": load_nict_tib1,
    "tibmd": load_tibmd,
}


def load_corpus(name: str, resample: bool = True) -> Manifest:
    if name not in CORPORA:
        raise ValueError(f"unknown corpus {name!r}; known: {sorted(CORPORA)}")
    return CORPORA[name](resample=resample)
