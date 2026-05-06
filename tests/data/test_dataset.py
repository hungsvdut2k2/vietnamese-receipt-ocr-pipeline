from pathlib import Path
import pytest

from vn_receipt_ocr.data.dataset import MCOCRDataset, group_lines_by_prefix


def test_group_lines_by_prefix(tmp_text_recognition_file: Path):
    grouped = group_lines_by_prefix(tmp_text_recognition_file)
    assert set(grouped.keys()) == {"img_a", "img_b", "img_c"}
    assert grouped["img_a"] == ["Hello", "World", "Foo"]
    assert grouped["img_b"] == ["One line only"]
    assert grouped["img_c"] == ["First", "Second", "Third"]  # sorted by suffix


def test_full_text_target_joins_with_newline(tmp_text_recognition_file: Path):
    ds = MCOCRDataset(
        text_path=tmp_text_recognition_file,
        images_dir=tmp_text_recognition_file.parent,
        instruction="X",
        require_images=False,
    )
    assert ds.full_text("img_a") == "Hello\nWorld\nFoo"
    assert ds.full_text("img_c") == "First\nSecond\nThird"


def test_dataset_len_equals_unique_prefixes(tmp_text_recognition_file: Path):
    ds = MCOCRDataset(
        text_path=tmp_text_recognition_file,
        images_dir=tmp_text_recognition_file.parent,
        instruction="X",
        require_images=False,
    )
    assert len(ds) == 3


def test_getitem_returns_dict_with_full_text_and_image_path(
    tmp_text_recognition_file: Path,
):
    ds = MCOCRDataset(
        text_path=tmp_text_recognition_file,
        images_dir=tmp_text_recognition_file.parent,
        instruction="X",
        require_images=False,
    )
    item = ds[0]
    assert "image_path" in item
    assert "full_text" in item
    assert "instruction" in item
    assert item["instruction"] == "X"


def test_missing_image_raises_when_required(tmp_text_recognition_file: Path):
    ds = MCOCRDataset(
        text_path=tmp_text_recognition_file,
        images_dir=tmp_text_recognition_file.parent,
        instruction="X",
        require_images=True,
    )
    with pytest.raises(FileNotFoundError):
        _ = ds[0]
