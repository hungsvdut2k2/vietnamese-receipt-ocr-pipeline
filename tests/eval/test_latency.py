from vn_receipt_ocr.eval.latency import compute_latency_percentiles


def test_p50_p95():
    times_ms = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    out = compute_latency_percentiles(times_ms)
    assert abs(out["p50"] - 55.0) < 1e-6
    assert abs(out["p95"] - 95.5) < 1.0  # numpy interpolation tolerance
