from __future__ import annotations
from typing import Any
from vn_receipt_ocr.config.models import ModelConfig, LoRAConfig, GPUProfileConfig


_DTYPE_TO_TORCH = {"fp16": "float16", "bf16": "bfloat16", "fp32": "float32"}


def build_load_kwargs(
    *, model: ModelConfig, gpu_profile: GPUProfileConfig
) -> dict[str, Any]:
    dtype_choice = model.dtype_override or gpu_profile.dtype
    return {
        "model_name": model.model_id,
        "dtype": _DTYPE_TO_TORCH[dtype_choice],
        "load_in_4bit": False,
    }


def build_peft_kwargs(*, lora: LoRAConfig) -> dict[str, Any]:
    return {
        "r": lora.rank,
        "lora_alpha": lora.alpha,
        "lora_dropout": lora.dropout,
        "target_modules": lora.target_modules,
        "bias": lora.bias,
        "finetune_vision_layers": lora.finetune_vision_layers,
        "finetune_language_layers": True,
        "finetune_attention_modules": True,
        "finetune_mlp_modules": True,
    }


def load_model_and_processor(
    *, model: ModelConfig, gpu_profile: GPUProfileConfig, lora: LoRAConfig
):
    """Load Qwen-VL with LoRA adapters via Unsloth.

    Heavy I/O; not tested in CI. Validates configuration via build_*_kwargs.
    """
    from unsloth import FastVisionModel  # lazy import — heavy

    load_kwargs = build_load_kwargs(model=model, gpu_profile=gpu_profile)
    base_model, processor = FastVisionModel.from_pretrained(**load_kwargs)
    peft_kwargs = build_peft_kwargs(lora=lora)
    peft_model = FastVisionModel.get_peft_model(base_model, **peft_kwargs)
    FastVisionModel.for_training(peft_model)
    return peft_model, processor
