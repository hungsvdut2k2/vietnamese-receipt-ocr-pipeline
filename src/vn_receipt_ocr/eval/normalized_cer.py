from __future__ import annotations

import re

from jiwer import cer as _jiwer_cer


_THOUSANDS_SEP_RE = re.compile(r"(?<=\d)[,.\s](?=\d{3}(?:[^\d]|$))")
_CURRENCY_SUFFIX_RE = re.compile(
    r"\s*(?:đồng|VNĐ|VND|đ|₫)\b|\s*(?:đồng|VNĐ|VND|đ|₫)$",
    flags=re.IGNORECASE,
)


def canonicalize_numeric(s: str) -> str:
    """Strip thousand separators (.,space) between digit groups."""
    prev = None
    cur = s
    while prev != cur:
        prev = cur
        cur = _THOUSANDS_SEP_RE.sub("", cur)
    return cur


def canonicalize_currency(s: str) -> str:
    """Strip Vietnamese currency suffixes."""
    # Run multiple times because suffixes may stack (e.g. "8600 VND đ").
    prev = None
    cur = s
    while prev != cur:
        prev = cur
        cur = _CURRENCY_SUFFIX_RE.sub("", cur)
    return cur.rstrip()


def normalize_for_cer(s: str) -> str:
    return canonicalize_numeric(canonicalize_currency(s))


def compute_normalized_cer(
    *, predictions: list[str], references: list[str]
) -> float:
    if len(predictions) != len(references):
        raise ValueError(
            f"length mismatch: predictions={len(predictions)} references={len(references)}"
        )
    p = [normalize_for_cer(x) for x in predictions]
    r = [normalize_for_cer(x) for x in references]
    return float(_jiwer_cer(r, p))
