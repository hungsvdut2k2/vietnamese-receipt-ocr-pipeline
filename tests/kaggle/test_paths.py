from pathlib import Path
from vn_receipt_ocr.kaggle.paths import resolve_dataset_path


def test_local_path_returned_when_kaggle_input_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("vn_receipt_ocr.kaggle.paths.KAGGLE_INPUT", tmp_path / "nope")
    local = tmp_path / "datasets" / "x"
    local.mkdir(parents=True)
    out = resolve_dataset_path(kaggle_subpath="x", local_fallback=str(local))
    assert out == local


def test_kaggle_input_path_used_when_present(tmp_path: Path, monkeypatch):
    kag = tmp_path / "kag_input"
    (kag / "x").mkdir(parents=True)
    monkeypatch.setattr("vn_receipt_ocr.kaggle.paths.KAGGLE_INPUT", kag)
    out = resolve_dataset_path(kaggle_subpath="x", local_fallback="/nonexistent")
    assert out == kag / "x"
