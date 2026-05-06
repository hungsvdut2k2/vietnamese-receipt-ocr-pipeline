from vn_receipt_ocr.eval.edit_ops import compute_edit_ops


def test_edit_ops_pure_substitution():
    ops = compute_edit_ops(predictions=["abc"], references=["abd"])
    assert ops == {"sub": 1, "ins": 0, "del": 0}


def test_edit_ops_pure_insertion():
    ops = compute_edit_ops(predictions=["abcd"], references=["abc"])
    assert ops == {"sub": 0, "ins": 1, "del": 0}


def test_edit_ops_pure_deletion():
    ops = compute_edit_ops(predictions=["ab"], references=["abc"])
    assert ops == {"sub": 0, "ins": 0, "del": 1}


def test_edit_ops_aggregates():
    ops = compute_edit_ops(
        predictions=["abc", "abcd"],
        references=["abd", "abc"],
    )
    assert ops == {"sub": 1, "ins": 1, "del": 0}
