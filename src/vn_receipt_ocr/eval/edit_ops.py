from __future__ import annotations

import jiwer


def compute_edit_ops(
    *, predictions: list[str], references: list[str]
) -> dict[str, int]:
    if len(predictions) != len(references):
        raise ValueError(
            f"length mismatch: predictions={len(predictions)} references={len(references)}"
        )
    out = jiwer.process_characters(references, predictions)
    return {
        "sub": int(out.substitutions),
        "ins": int(out.insertions),
        "del": int(out.deletions),
    }
