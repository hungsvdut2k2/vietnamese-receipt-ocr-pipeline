from PIL import Image
import pytest

from vn_receipt_ocr.data.prompt import PromptBuilder


@pytest.fixture
def dummy_image() -> Image.Image:
    return Image.new("RGB", (100, 100), color="white")


def test_build_messages_user_then_assistant(dummy_image):
    pb = PromptBuilder(instruction="Trích xuất tất cả nội dung.")
    msgs = pb.build_train_messages(image=dummy_image, target="Hello\nWorld")
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


def test_user_content_has_image_and_text(dummy_image):
    pb = PromptBuilder(instruction="X")
    msgs = pb.build_train_messages(image=dummy_image, target="t")
    user_content = msgs[0]["content"]
    types = [c["type"] for c in user_content]
    assert types == ["image", "text"]
    assert user_content[1]["text"] == "X"


def test_assistant_content_is_target(dummy_image):
    pb = PromptBuilder(instruction="X")
    msgs = pb.build_train_messages(image=dummy_image, target="t1\nt2")
    assert msgs[1]["content"] == "t1\nt2"


def test_inference_messages_omit_assistant(dummy_image):
    pb = PromptBuilder(instruction="X")
    msgs = pb.build_inference_messages(image=dummy_image)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
