from __future__ import annotations
import time
import torch

from vn_receipt_ocr.data.image_io import load_image_rgb
from vn_receipt_ocr.data.prompt import PromptBuilder


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
    """Greedy decode for each item. Returns (predictions, latency_ms_per_sample).

    Latency is the batch's wall-clock generate time divided by len(batch);
    every sample in a batch reports the same value. For true per-item
    percentiles, run with batch_size=1.
    """
    pb = PromptBuilder(instruction=instruction)
    preds: list[str] = []
    times: list[float] = []
    for start in range(0, len(items), batch_size):
        batch = items[start : start + batch_size]
        imgs = [load_image_rgb(it["image_path"]) for it in batch]
        texts = [
            processor.apply_chat_template(
                pb.build_inference_messages(image=im),
                tokenize=False, add_generation_prompt=True,
            )
            for im in imgs
        ]
        inputs = processor(text=texts, images=imgs, return_tensors="pt",
                           padding=True).to(model.device)
        t0 = time.perf_counter()
        out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                             do_sample=False)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0 / len(batch)
        # Slice off the prompt tokens to get only the generated suffix.
        gen = out[:, inputs["input_ids"].shape[1]:]
        decoded = processor.batch_decode(gen, skip_special_tokens=True)
        preds.extend(decoded)
        times.extend([elapsed_ms] * len(batch))
    return preds, times
