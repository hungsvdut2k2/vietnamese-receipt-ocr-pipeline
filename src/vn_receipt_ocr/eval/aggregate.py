from __future__ import annotations

from vn_receipt_ocr.eval.cer import compute_cer
from vn_receipt_ocr.eval.diacritic_cer import compute_diacritic_cer
from vn_receipt_ocr.eval.edit_ops import compute_edit_ops
from vn_receipt_ocr.eval.latency import compute_latency_percentiles
from vn_receipt_ocr.eval.length_ratio import compute_length_ratios
from vn_receipt_ocr.eval.normalized_cer import compute_normalized_cer
from vn_receipt_ocr.eval.wer import compute_wer


def aggregate_metrics(
    *,
    predictions: list[str],
    references: list[str],
    latency_ms: list[float],
    metrics_enabled: list[str],
) -> dict:
    out: dict = {}
    if "cer" in metrics_enabled:
        out["cer"] = compute_cer(predictions=predictions, references=references)
    if "diacritic_cer" in metrics_enabled:
        out["diacritic_cer"] = compute_diacritic_cer(
            predictions=predictions, references=references
        )
    if "cer_normalized" in metrics_enabled:
        out["cer_normalized"] = compute_normalized_cer(
            predictions=predictions, references=references
        )
    if "wer" in metrics_enabled:
        out["wer"] = compute_wer(predictions=predictions, references=references)
    if "edit_ops" in metrics_enabled:
        out["edit_ops"] = compute_edit_ops(
            predictions=predictions, references=references
        )
    if "length_ratio" in metrics_enabled:
        ratios = compute_length_ratios(
            predictions=predictions, references=references
        )
        finite = [r for r in ratios if r != float("inf")]
        out["mean_length_ratio"] = (sum(finite) / len(finite)) if finite else 0.0
        out["length_ratios"] = ratios
    if "empty_pred_rate" in metrics_enabled:
        if predictions:
            out["empty_pred_rate"] = sum(1 for p in predictions if len(p) == 0) / len(predictions)
        else:
            out["empty_pred_rate"] = 0.0
    if "latency" in metrics_enabled:
        pct = compute_latency_percentiles(latency_ms)
        out["latency_p50"] = pct["p50"]
        out["latency_p95"] = pct["p95"]
    return out
