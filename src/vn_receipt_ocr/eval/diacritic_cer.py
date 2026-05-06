from __future__ import annotations

import unicodedata

from jiwer import cer as _jiwer_cer

VIETNAMESE_DIACRITIC_MARKS = {
    "̀",  # combining grave accent
    "́",  # combining acute accent
    "̂",  # combining circumflex accent
    "̃",  # combining tilde
    "̆",  # combining breve
    "̉",  # combining hook above
    "̛",  # combining horn
    "̣",  # combining dot below
}


def _has_vietnamese_diacritic(ch: str) -> bool:
    decomposed = unicodedata.normalize("NFD", ch)
    return any(c in VIETNAMESE_DIACRITIC_MARKS for c in decomposed)


def filter_to_diacritic_chars(s: str) -> str:
    return "".join(ch for ch in s if _has_vietnamese_diacritic(ch))


def compute_diacritic_cer(
    *, predictions: list[str], references: list[str]
) -> float:
    if len(predictions) != len(references):
        raise ValueError("predictions and references length mismatch")
    p_filt = [filter_to_diacritic_chars(p) for p in predictions]
    r_filt = [filter_to_diacritic_chars(r) for r in references]
    # If no diacritics anywhere in references, return 0.0 (sentinel)
    if all(len(r) == 0 for r in r_filt):
        return 0.0
    # jiwer cer accepts empty strings as long as not ALL are empty
    return float(_jiwer_cer(r_filt, p_filt))
