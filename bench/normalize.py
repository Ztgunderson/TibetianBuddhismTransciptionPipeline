"""Per-language text normalization for error-rate scoring.

WHY THIS FILE IS LOAD-BEARING
-----------------------------
The headline result (Loop 1) is a *difference in error rates between languages*. If Tibetan
text is normalized more or less aggressively than English, that difference is manufactured by
the normalizer, not by the model. So every normalization decision here is explicit, documented,
and identical in spirit across languages: lowercase-equivalent folding, punctuation stripping,
whitespace collapse — nothing that removes linguistic content.

Tibetan specifics:
  - TSHEG (U+0F0B, ་) is the morpheme/syllable delimiter — the "space" of Tibetan. It is NOT
    punctuation to be deleted; for CER we keep syllable content but normalize the delimiter so
    a model that emits a space vs. a tsheg isn't penalized for orthographic style. We fold
    tsheg -> tsheg (kept) and only collapse runs.
  - SHAD (U+0F0D, ། and U+0F0E, ༎) is the clause/sentence terminator — closer to a period.
    We strip it for CER the way we strip '.' for English, symmetrically.
  - We do NOT apply Unicode NFKC blindly: some NFKC mappings alter Tibetan stacks. We use NFC,
    which is the canonical composed form and safe for Tibetan.

Sanskrit (Devanagari) specifics:
  - DANDA (U+0964, ।) and DOUBLE DANDA (U+0965, ॥) are sentence terminators — stripped like
    English '.', symmetrically.
  - AVAGRAHA and combining marks are linguistic content and kept.

The guiding rule: whatever we remove for one language, we remove the analogous thing for the
others. Punctuation-and-spacing only, never content.
"""

from __future__ import annotations

import re
import unicodedata

# Tibetan
TSHEG = "་"          # ་ syllable delimiter (kept as content boundary)
TIB_TERMINATORS = "།༎༑༒༔"  # shad and relatives (stripped, like '.')

# Devanagari (Sanskrit)
DEV_TERMINATORS = "।॥"  # danda, double danda (stripped, like '.')

# Latin punctuation stripped for English (symmetric with the terminators above)
_LATIN_PUNCT = re.compile(r"[.,!?;:\"'`()\[\]{}<>«»…–—-]")
_WS = re.compile(r"\s+")


def _nfc(text: str) -> str:
    return unicodedata.normalize("NFC", text)


def normalize_english(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = _nfc(text).lower()
    text = _LATIN_PUNCT.sub(" ", text)
    return _WS.sub(" ", text).strip()


def normalize_tibetan(text: str) -> str:
    """Strip sentence terminators (shad); normalize tsheg runs; collapse whitespace.

    Tsheg is kept because it delimits syllables (content structure); we only collapse
    repeated tsheg/space runs to a single tsheg so decode-time spacing style is not scored.
    """
    text = _nfc(text)
    for ch in TIB_TERMINATORS:
        text = text.replace(ch, " ")
    # Treat ASCII space and tsheg as the same delimiter, then re-emit a single tsheg.
    text = text.replace(" ", TSHEG)
    text = re.sub(TSHEG + "+", TSHEG, text)
    return text.strip(TSHEG + " ")


def normalize_sanskrit(text: str) -> str:
    """Strip danda/double-danda; collapse whitespace. Devanagari combining marks kept."""
    text = _nfc(text)
    for ch in DEV_TERMINATORS:
        text = text.replace(ch, " ")
    return _WS.sub(" ", text).strip()


_NORMALIZERS = {
    "en": normalize_english,
    "bo": normalize_tibetan,
    "sa": normalize_sanskrit,
}


def normalize(text: str, lang: str) -> str:
    """Dispatch to the language-appropriate normalizer. Unknown langs get whitespace-collapse only."""
    fn = _NORMALIZERS.get(lang)
    if fn is None:
        return _WS.sub(" ", _nfc(text)).strip()
    return fn(text)
