from vn_receipt_ocr.eval.wer import compute_wer


def test_wer_zero_for_identical():
    assert compute_wer(predictions=["a b c"], references=["a b c"]) == 0.0


def test_wer_one_for_completely_wrong():
    assert compute_wer(predictions=["x y z"], references=["a b c"]) == 1.0


def test_wer_aggregates_across_corpus():
    wer = compute_wer(
        predictions=["a b c", "a b c d"],
        references=["a b c", "a b c e"],
    )
    # total ref words = 7; total errors = 1 (substitution in second pair)
    assert abs(wer - 1 / 7) < 1e-6
