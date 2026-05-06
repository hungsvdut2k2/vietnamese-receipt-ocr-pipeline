from __future__ import annotations


def compute_length_ratios(
    *, predictions: list[str], references: list[str]
) -> list[float]:
    if len(predictions) != len(references):
        raise ValueError(
            f"length mismatch: predictions={len(predictions)} references={len(references)}"
        )
    out: list[float] = []
    for p, r in zip(predictions, references):
        if len(r) == 0:
            out.append(float("inf"))
        else:
            out.append(len(p) / len(r))
    return out
