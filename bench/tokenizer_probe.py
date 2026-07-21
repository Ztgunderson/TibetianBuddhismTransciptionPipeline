"""Loop 1b — token-level sparsity probe.

Measures how many decoder tokens each language costs *per unit of written content* under
Whisper's multilingual BPE tokenizer. Runs on the CPU-only host — no GPU, no model inference,
just tokenization — so it is the one Loop 1 measurement available before the model ladder is built.

WHY THIS IS A REAL FINDING, NOT PLUMBING
----------------------------------------
The survey (arXiv 2510.19144) attributes Tibetan's failure to "token-level sparsity and structural
underrepresentation" but never measures it. Whisper's BPE was learned on a mostly-English corpus,
so common English spans merge into single tokens while Tibetan script — barely present in the
merge table — falls back toward raw UTF-8 bytes. The result: the *same meaning* costs Tibetan far
more autoregressive decoding steps, and every extra step is a place an ASR error can be introduced
or compounded. This gives a mechanistic account of part of H1 (the quantization tax): a decoder
that must emit ~Nx more tokens for Tibetan has ~Nx more opportunities to derail once quantization
degrades it.

Reported per language:
  chars/token   — higher is more efficient (English high, Tibetan low)
  tokens/char   — the inverse; the "step multiplier" vs. a high-resource language
  bytes/token   — near 1.0 indicates byte-fallback (no useful merges for that script)

Run:
  python3 -m bench.tokenizer_probe                 # built-in illustrative samples
  python3 -m bench.tokenizer_probe --manifest ...  # on real corpus references (preferred)
"""

from __future__ import annotations

import argparse
import json
import statistics
from dataclasses import dataclass

# Illustrative samples — short, representative sentences per language. These are for a runnable
# demonstration TODAY; the reported paper number comes from --manifest over the real corpora,
# where per-utterance distributions (not just point values) are summarized.
SAMPLES = {
    "en": [
        "the nature of mind is clear and knowing",
        "all phenomena arise from causes and conditions",
        "meditate on impermanence every single day",
    ],
    "bo": [
        "སེམས་ཀྱི་རང་བཞིན་ནི་གསལ་ཞིང་རིག་པ་ཡིན",
        "ཆོས་ཐམས་ཅད་རྒྱུ་རྐྱེན་ལས་བྱུང་བ་ཡིན",
        "ཉིན་རེ་བཞིན་མི་རྟག་པ་བསྒོམ་པར་བྱའོ",
    ],
    "sa": [
        "सर्वं क्षणिकं दुःखम् अनात्मकं च",
        "हेतुप्रत्ययसमुत्पन्नाः सर्वधर्माः",
        "प्रतिदिनम् अनित्यताम् भावयेत्",
    ],
}


@dataclass
class LangStat:
    lang: str
    n: int
    chars_per_token: float
    tokens_per_char: float
    bytes_per_token: float

    def __str__(self) -> str:
        return (f"  {self.lang}:  {self.chars_per_token:6.2f} chars/tok   "
                f"{self.tokens_per_char:6.2f} tok/char   "
                f"{self.bytes_per_token:5.2f} bytes/tok   (n={self.n})")


def _get_tokenizer():
    from whisper.tokenizer import get_tokenizer
    return get_tokenizer(multilingual=True).encoding


def measure(texts_by_lang: dict[str, list[str]]) -> dict[str, LangStat]:
    enc = _get_tokenizer()
    out: dict[str, LangStat] = {}
    for lang, texts in texts_by_lang.items():
        cpt, tpc, bpt = [], [], []
        for t in texts:
            if not t.strip():
                continue
            n_tok = len(enc.encode(t))
            n_char = len(t)
            n_byte = len(t.encode("utf-8"))
            if n_tok == 0:
                continue
            cpt.append(n_char / n_tok)
            tpc.append(n_tok / n_char)
            bpt.append(n_byte / n_tok)
        out[lang] = LangStat(
            lang, len(cpt),
            statistics.mean(cpt), statistics.mean(tpc), statistics.mean(bpt),
        )
    return out


def _load_manifest(path: str) -> dict[str, list[str]]:
    by_lang: dict[str, list[str]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                r = json.loads(line)
                by_lang.setdefault(r["lang"], []).append(r["reference"])
    return by_lang


def main() -> None:
    ap = argparse.ArgumentParser(description="Whisper tokenizer sparsity probe (Loop 1b).")
    ap.add_argument("--manifest", help="JSONL manifest with 'lang' and 'reference' fields")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    data = _load_manifest(args.manifest) if args.manifest else SAMPLES
    stats = measure(data)

    if args.json:
        print(json.dumps({k: vars(v) for k, v in stats.items()}, ensure_ascii=False, indent=2))
        return

    print("Whisper BPE token cost per language "
          f"({'corpus' if args.manifest else 'illustrative samples'}):")
    for s in stats.values():
        print(s)
    if "en" in stats:
        base = stats["en"].tokens_per_char
        print("\nStep multiplier vs. English (tokens per char, higher = more decoder steps):")
        for lang, s in stats.items():
            print(f"  {lang}: {s.tokens_per_char / base:5.1f}x")


if __name__ == "__main__":
    main()
