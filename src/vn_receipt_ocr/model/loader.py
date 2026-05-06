from __future__ import annotations
import re

import torch

from vn_receipt_ocr.config.models import ModelConfig, LoRAConfig, GPUProfileConfig


_DTYPE_TO_TORCH = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def resolve_dtype(
    *, model: ModelConfig, gpu_profile: GPUProfileConfig
) -> torch.dtype:
    return _DTYPE_TO_TORCH[model.dtype_override or gpu_profile.dtype]


# Matches the vision branch under any common Qwen-VL naming. Used to freeze
# vision parameters (and any LoRA adapters peft attached inside the vision
# tower by suffix-matching target_modules across the whole model).
_VISION_PATH = re.compile(r"(?:^|\.)(visual|vision_tower|vision_model)\.")


def load_model_and_processor(
    *, model: ModelConfig, gpu_profile: GPUProfileConfig, lora: LoRAConfig
):
    """Load Qwen-VL with a LoRA adapter via stock transformers + peft.

    Heavy I/O; not exercised in CI.
    """
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from peft import LoraConfig, get_peft_model

    torch_dtype = resolve_dtype(model=model, gpu_profile=gpu_profile)

    processor = AutoProcessor.from_pretrained(model.model_id)
    base = AutoModelForImageTextToText.from_pretrained(
        model.model_id,
        torch_dtype=torch_dtype,
        device_map="auto",
    )
    # Required for grad-checkpointing + LoRA: makes embedding outputs
    # require_grad so backprop reaches the adapters.
    base.enable_input_require_grads()

    peft_config = LoraConfig(
        r=lora.rank,
        lora_alpha=lora.alpha,
        lora_dropout=lora.dropout,
        target_modules=list(lora.target_modules),
        bias=lora.bias,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(base, peft_config)

    if model.freeze_vision_tower or not lora.finetune_vision_layers:
        for name, param in peft_model.named_parameters():
            if _VISION_PATH.search(name):
                param.requires_grad = False

    peft_model.train()
    return peft_model, processor
