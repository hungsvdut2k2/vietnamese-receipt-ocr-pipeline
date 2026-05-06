import subprocess
import sys


def test_help_runs():
    out = subprocess.run(
        [sys.executable, "-m", "vn_receipt_ocr", "--help"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0
    assert "train" in out.stdout
    assert "eval" in out.stdout
    assert "predict" in out.stdout


def test_train_help_runs():
    out = subprocess.run(
        [sys.executable, "-m", "vn_receipt_ocr", "train", "--help"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0
    assert "--config" in out.stdout


def test_top_level_imports():
    import vn_receipt_ocr as m
    assert hasattr(m, "train")
    assert hasattr(m, "evaluate")
    assert hasattr(m, "predict")
    assert hasattr(m, "TrainConfig")
