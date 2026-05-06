import torch
from PIL import Image

from vn_receipt_ocr.data.collator import QwenVLCollator


class _FakeProcessor:
    """Minimal processor that mimics Qwen-VL's interface for collator tests."""

    def __init__(self):
        self.tokenizer = self  # for parity with HF processors
        self.pad_token_id = 0

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        # Concatenate all 'text' fields into a string the test can grep for.
        parts = []
        for m in messages:
            if isinstance(m["content"], list):
                for c in m["content"]:
                    if c["type"] == "text":
                        parts.append(c["text"])
            else:
                parts.append(m["content"])
        return f"<|{m['role']}|>" + " ".join(parts)

    def __call__(self, text, images=None, return_tensors="pt", padding=True):
        # Build small int tensors of size = len(text)*5 per example.
        batch = {}
        ids = [list(range(len(t) % 7 + 3)) for t in text]
        max_len = max(len(x) for x in ids)
        padded = [x + [self.pad_token_id] * (max_len - len(x)) for x in ids]
        attn = [[1]*len(x) + [0]*(max_len - len(x)) for x in ids]
        batch["input_ids"] = torch.tensor(padded)
        batch["attention_mask"] = torch.tensor(attn)
        batch["pixel_values"] = torch.zeros(len(text), 3, 8, 8)
        return batch


def test_collator_masks_user_tokens_with_neg100():
    proc = _FakeProcessor()
    coll = QwenVLCollator(processor=proc)
    img = Image.new("RGB", (32, 32))
    items = [
        {"image_path": img, "instruction": "Q", "full_text": "A"},
        {"image_path": img, "instruction": "Q", "full_text": "AA"},
    ]
    batch = coll(items)
    # Labels exist and have same shape as input_ids
    assert "labels" in batch
    assert batch["labels"].shape == batch["input_ids"].shape
    # At least one position is masked to -100
    assert (batch["labels"] == -100).any()


def test_collator_handles_pil_image_objects(monkeypatch):
    proc = _FakeProcessor()
    coll = QwenVLCollator(processor=proc)
    img = Image.new("RGB", (32, 32))
    items = [{"image_path": img, "instruction": "Q", "full_text": "A"}]
    batch = coll(items)
    assert batch["pixel_values"].shape[0] == 1
