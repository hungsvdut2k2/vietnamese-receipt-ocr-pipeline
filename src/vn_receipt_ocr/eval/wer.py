from __future__ import annotations

from jiwer import wer


def compute_wer(*, predictions: list[str], references: list[str]) -> float:
    if not predictions or not references:
        raise ValueError("predictions and references must be non-empty")
    if len(predictions) != len(references):
        raise ValueError(
            f"length mismatch: predictions={len(predictions)} references={len(references)}"
        )
    return float(wer(references, predictions))
