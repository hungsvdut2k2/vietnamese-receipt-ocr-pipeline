from __future__ import annotations

import numpy as np


def compute_latency_percentiles(times_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(times_ms, dtype=float)
    if arr.size == 0:
        return {"p50": 0.0, "p95": 0.0}
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }
