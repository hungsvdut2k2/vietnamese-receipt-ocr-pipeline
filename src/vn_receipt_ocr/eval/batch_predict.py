from __future__ import annotations
import time
import torch
from PIL import Image

from vn_receipt_ocr.data.prompt import PromptBuilder


def _load_image(p) -> Image.Image:
    if isinstance(p, Image.Image):
        return p.convert("RGB")
    return Image.open(p).convert("RGB")


@torch.inference_mode()
def batch_predict(
    *,
    model,
    processor,
    items: list[dict],
    instruction: str,
    max_new_tokens: int = 244,
    batch_size: int = 1,
) -> tuple[list[str], list[float]]:
    """Greedy decode for each item. Returns (predictions, latency_ms_per_sample)."""
    pb = PromptBuilder(instruction=instruction)
    preds: list[str] = []
    times: list[float] = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        imgs = [_load_image(it["image_path"]) for it in batch]
        texts = [
            processor.apply_chat_template(
                pb.build_inference_messages(image=im),
                tokenize=False, add_generation_prompt=True,
            )
            for im in imgs
        ]
        inputs = processor(text=texts, images=imgs, return_tensors="pt",
                           padding=True).to(model.device)
        t0 = time.time()
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False)
        elapsed_ms = (time.time() - t0) * 1000.0 / len(batch)
        # Slice off the prompt tokens to get only the generated suffix.
        gen = out[:, inputs["input_ids"].shape[1]:]
        decoded = processor.batch_decode(gen, skip_special_tokens=True)
        preds.extend(decoded)
        times.extend([elapsed_ms] * len(batch))
    return preds, times
