from vn_receipt_ocr.model.loader import build_load_kwargs, build_peft_kwargs
from vn_receipt_ocr.config.models import ModelConfig, LoRAConfig, GPUProfileConfig


def test_load_kwargs_uses_gpu_profile_dtype():
    kw = build_load_kwargs(
        model=ModelConfig(model_id="x"),
        gpu_profile=GPUProfileConfig(name="p100_16gb", dtype="fp16",
                                     vram_gb=16, recommended_batch_size=1),
    )
    assert kw["model_name"] == "x"
    assert kw["dtype"] == "float16"
    assert kw["load_in_4bit"] is False


def test_load_kwargs_dtype_override_wins():
    kw = build_load_kwargs(
        model=ModelConfig(model_id="x", dtype_override="bf16"),
        gpu_profile=GPUProfileConfig(name="t4_16gb", dtype="fp16",
                                     vram_gb=16, recommended_batch_size=1),
    )
    assert kw["dtype"] == "bfloat16"


def test_peft_kwargs_freeze_vision():
    kw = build_peft_kwargs(
        lora=LoRAConfig(rank=16, alpha=32, dropout=0.05,
                        target_modules=["q_proj","k_proj","v_proj"],
                        bias="none", finetune_vision_layers=False),
    )
    assert kw["r"] == 16
    assert kw["lora_alpha"] == 32
    assert kw["lora_dropout"] == 0.05
    assert kw["target_modules"] == ["q_proj","k_proj","v_proj"]
    assert kw["bias"] == "none"
    assert kw["finetune_vision_layers"] is False
    assert kw["finetune_language_layers"] is True
    assert kw["finetune_attention_modules"] is True
    assert kw["finetune_mlp_modules"] is True
