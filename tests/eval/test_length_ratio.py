from vn_receipt_ocr.eval.length_ratio import compute_length_ratios


def test_length_ratios():
    rs = compute_length_ratios(predictions=["abc", "ab"], references=["abc", "abc"])
    assert rs == [1.0, 2/3]


def test_length_ratio_zero_reference_yields_inf():
    rs = compute_length_ratios(predictions=["abc"], references=[""])
    assert rs[0] == float("inf")
