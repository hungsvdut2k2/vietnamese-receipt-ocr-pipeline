from vn_receipt_ocr.eval.normalized_cer import (
    canonicalize_numeric,
    canonicalize_currency,
    compute_normalized_cer,
)


def test_canonicalize_numeric_strips_thousands_separators():
    assert canonicalize_numeric("8,600") == "8600"
    assert canonicalize_numeric("8.600") == "8600"
    assert canonicalize_numeric("1,234,567.89") == "1234567.89"


def test_canonicalize_currency_strips_suffix():
    assert canonicalize_currency("8600đ") == "8600"
    assert canonicalize_currency("8600 VND") == "8600"
    assert canonicalize_currency("8600đồng") == "8600"


def test_normalized_cer_treats_format_variants_as_equal():
    # Two strings differ only in numeric format and currency suffix
    cer = compute_normalized_cer(
        predictions=["Tổng: 8,600đ"],
        references=["Tổng: 8600"],
    )
    assert cer == 0.0
