# tests/conftest.py
from pathlib import Path
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def mcocr_root(repo_root: Path) -> Path:
    return repo_root / "datasets" / "kagglehub" / "datasets" / "domixi1989" \
        / "vietnamese-receipts-mc-ocr-2021" / "versions" / "17"


@pytest.fixture
def tmp_text_recognition_file(tmp_path: Path) -> Path:
    """Tiny hand-crafted line-OCR file for deterministic dataset tests."""
    f = tmp_path / "tiny.txt"
    f.write_text(
        "img_a_0.jpg\tHello\n"
        "img_a_1.jpg\tWorld\n"
        "img_a_2.jpg\tFoo\n"
        "img_b_0.jpg\tOne line only\n"
        "img_c_2.jpg\tThird\n"  # out-of-order suffix
        "img_c_0.jpg\tFirst\n"
        "img_c_1.jpg\tSecond\n",
        encoding="utf-8",
    )
    return f
