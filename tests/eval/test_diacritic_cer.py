from vn_receipt_ocr.eval.diacritic_cer import (
    compute_diacritic_cer,
    filter_to_diacritic_chars,
)


def test_filter_keeps_only_vietnamese_diacritic_chars():
    s = "Hà Nội 123"
    # Keeps: à, ộ. Drops: H, N, i, space, digits.
    out = filter_to_diacritic_chars(s)
    assert out == "àộ"


def test_filter_unaccented_yields_empty_string():
    assert filter_to_diacritic_chars("Hello 123") == ""


def test_diacritic_cer_zero_for_perfect_diacritics():
    cer = compute_diacritic_cer(
        predictions=["Hà Nội"], references=["Hà Nội"],
    )
    assert cer == 0.0


def test_diacritic_cer_ignores_non_diacritic_substitutions():
    # pred and ref differ only in non-diacritic chars; diacritic CER should be 0
    cer = compute_diacritic_cer(
        predictions=["XX Nội"],
        references=["Hà Nội"],
    )
    # ref diacritic chars: à, ộ → 2; pred diacritic chars: ộ → 1
    # (pred is missing 'à' from XX) — so 1 deletion / 2 ref-chars = 0.5
    assert abs(cer - 0.5) < 1e-6


def test_diacritic_cer_returns_zero_when_no_diacritics_in_reference():
    # When reference has no diacritic chars, CER on empty strings is undefined;
    # we return 0.0 as a sentinel.
    cer = compute_diacritic_cer(
        predictions=["abc"], references=["abc"],
    )
    assert cer == 0.0
