from __future__ import annotations

from typing import Any

import torch
from PIL import Image

from vn_receipt_ocr.data.prompt import PromptBuilder


def _load_image(image_path) -> Image.Image:
    if isinstance(image_path, Image.Image):
        return image_path.convert("RGB")
    return Image.open(image_path).convert("RGB")


class QwenVLCollator:
    """Build a training batch for Qwen-VL SFT.

    For each item, render full chat template (user + assistant) and a
    user-only template (no assistant). Tokenize both; mask positions in
    `labels` that fall within the user-only prefix to -100, so loss is
    response-only.
    """

    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def _build_prompt_builder(self, instruction: str) -> PromptBuilder:
        return PromptBuilder(instruction=instruction)

    def __call__(self, items: list[dict]) -> dict[str, torch.Tensor]:
        full_texts: list[str] = []
        prefix_only_texts: list[str] = []
        images: list[Image.Image] = []

        for it in items:
            pb = self._build_prompt_builder(it["instruction"])
            img = _load_image(it["image_path"])
            images.append(img)
            full_msgs = pb.build_train_messages(image=img, target=it["full_text"])
            prefix_msgs = pb.build_inference_messages(image=img)
            full_texts.append(self.processor.apply_chat_template(
                full_msgs, tokenize=False, add_generation_prompt=False))
            prefix_only_texts.append(self.processor.apply_chat_template(
                prefix_msgs, tokenize=False, add_generation_prompt=True))

        full = self.processor(text=full_texts, images=images,
                              return_tensors="pt", padding=True)
        prefix = self.processor(text=prefix_only_texts, images=images,
                                return_tensors="pt", padding=True)

        labels = full["input_ids"].clone()
        for i in range(labels.shape[0]):
            prefix_len = int(prefix["attention_mask"][i].sum().item())
            labels[i, :prefix_len] = -100
        labels[full["attention_mask"] == 0] = -100

        full["labels"] = labels
        return full
