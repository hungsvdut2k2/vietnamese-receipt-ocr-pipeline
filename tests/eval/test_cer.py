from vn_receipt_ocr.eval.cer import compute_cer


def test_cer_zero_for_identical_strings():
    assert compute_cer(predictions=["abc"], references=["abc"]) == 0.0


def test_cer_one_for_completely_wrong():
    # 3-char ref, 3 substitutions
    assert compute_cer(predictions=["xyz"], references=["abc"]) == 1.0


def test_cer_aggregates_across_corpus():
    cer = compute_cer(
        predictions=["abc", "abcd"],
        references=["abc", "abce"],
    )
    # total ref chars = 7; total errors = 1 (substitution in second pair)
    assert abs(cer - 1 / 7) < 1e-6


def test_cer_empty_inputs_raises():
    import pytest
    with pytest.raises(ValueError):
        compute_cer(predictions=[], references=[])
