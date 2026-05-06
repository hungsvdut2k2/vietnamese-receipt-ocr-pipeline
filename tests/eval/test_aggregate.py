from vn_receipt_ocr.eval.aggregate import aggregate_metrics


def test_aggregate_all_metrics():
    out = aggregate_metrics(
        predictions=["Hà Nội", "abc"],
        references=["Hà Nội", "abc"],
        latency_ms=[10.0, 20.0],
        metrics_enabled=["cer","diacritic_cer","cer_normalized","wer",
                         "edit_ops","length_ratio","empty_pred_rate","latency"],
    )
    assert out["cer"] == 0.0
    assert out["diacritic_cer"] == 0.0
    assert out["cer_normalized"] == 0.0
    assert out["wer"] == 0.0
    assert out["edit_ops"]["sub"] == 0
    assert out["mean_length_ratio"] == 1.0
    assert out["empty_pred_rate"] == 0.0
    assert out["latency_p50"] == 15.0


def test_aggregate_empty_pred_rate():
    out = aggregate_metrics(
        predictions=["", "abc"],
        references=["xyz", "abc"],
        latency_ms=[10.0, 20.0],
        metrics_enabled=["empty_pred_rate"],
    )
    assert out["empty_pred_rate"] == 0.5
