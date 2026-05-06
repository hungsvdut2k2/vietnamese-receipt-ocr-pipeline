import pytest
from pydantic import ValidationError

from vn_receipt_ocr.config.models import (
    TrainConfig, ModelConfig, LoRAConfig, DataConfig, TrainerConfig,
    GPUProfileConfig, WandBConfig, HFHubConfig, EvalConfig,
)


def test_lora_config_rejects_negative_rank():
    with pytest.raises(ValidationError):
        LoRAConfig(rank=-1, alpha=32, dropout=0.0,
                   target_modules=["q_proj"], bias="none",
                   finetune_vision_layers=False)


def test_gpu_profile_rejects_unknown_dtype():
    with pytest.raises(ValidationError):
        GPUProfileConfig(name="x", dtype="int4", vram_gb=16,
                         recommended_batch_size=1)


def test_train_config_round_trip():
    cfg = TrainConfig(
        model=ModelConfig(model_id="Qwen/Qwen3-VL-2B-Instruct"),
        lora=LoRAConfig(rank=16, alpha=32, dropout=0.05,
                        target_modules=["q_proj","k_proj","v_proj","o_proj",
                                        "gate_proj","up_proj","down_proj"],
                        bias="none", finetune_vision_layers=False),
        data=DataConfig(
            train_path="datasets/.../text_recognition_train_data.txt",
            train_images_dir="datasets/.../train_images",
            val_path="datasets/.../text_recognition_val_data.txt",
            val_images_dir="datasets/.../val_images",
            instruction="Trích xuất...",
            target_resolution=888, max_resolution=1388, max_seq_length=203,
        ),
        trainer=TrainerConfig(
            epochs=3, per_device_batch_size=1, grad_accum=8, lr=1e-4,
            warmup_ratio=0.05, optimizer="adamw_8bit",
            gradient_checkpointing=True, seed=42, deterministic=False,
            wallclock_budget_hours=8.0,
        ),
        gpu_profile=GPUProfileConfig(
            name="p100_16gb", dtype="fp16", vram_gb=16,
            recommended_batch_size=1,
        ),
        wandb=WandBConfig(project="vn-receipt-ocr",
                          run_name_template="{model_short}-r{lora_rank}-{date}-{nnn}",
                          enabled=True, mode_fallback="offline"),
        hf_hub=HFHubConfig(repo_owner="me", repo_name_template="vn-receipt-ocr-{run_id}",
                           private=True, enabled=True),
        eval=EvalConfig(
            metrics_enabled=["cer","diacritic_cer","cer_normalized","wer",
                             "edit_ops","length_ratio","empty_pred_rate","latency"],
            decode="greedy", max_new_tokens=244, sample_table_size=10,
        ),
    )
    assert cfg.lora.rank == 16
    assert cfg.gpu_profile.dtype == "fp16"
