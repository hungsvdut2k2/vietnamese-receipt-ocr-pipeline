from vn_receipt_ocr.eval.wer import compute_wer


def test_wer_zero_for_identical():
    assert compute_wer(predictions=["a b c"], references=["a b c"]) == 0.0


def test_wer_one_for_completely_wrong():
    assert compute_wer(predictions=["x y z"], references=["a b c"]) == 1.0
