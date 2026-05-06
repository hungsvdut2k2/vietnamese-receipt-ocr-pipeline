from __future__ import annotations
from pathlib import Path  # noqa: F401
from typing import Literal
from pydantic import BaseModel, ConfigDict, Field, field_validator


_DTYPE = Literal["bf16", "fp16", "fp32"]
_OPTIM = Literal["adamw_8bit", "adamw_torch", "paged_adamw_8bit"]
_DECODE = Literal["greedy", "beam"]
_BIAS = Literal["none", "all", "lora_only"]


class ModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model_id: str = Field(min_length=1)
    dtype_override: _DTYPE | None = None        # if None, GPU profile decides
    freeze_vision_tower: bool = True


class LoRAConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rank: int = Field(gt=0)
    alpha: int = Field(gt=0)
    dropout: float = Field(ge=0.0, le=1.0)
    target_modules: list[str] = Field(min_length=1)
    bias: _BIAS = "none"
    finetune_vision_layers: bool = False


class DataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    train_path: str
    train_images_dir: str
    val_path: str
    val_images_dir: str
    test_path: str | None = None
    test_images_dir: str | None = None
    instruction: str = Field(min_length=1)
    target_resolution: int = Field(gt=0)
    max_resolution: int = Field(gt=0)
    max_seq_length: int = Field(gt=0)


class TrainerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    epochs: int = Field(gt=0)
    per_device_batch_size: int = Field(gt=0)
    grad_accum: int = Field(gt=0)
    lr: float = Field(gt=0)
    warmup_ratio: float = Field(ge=0.0, lt=1.0)
    optimizer: _OPTIM = "adamw_8bit"
    gradient_checkpointing: bool = True
    seed: int = 42
    deterministic: bool = False
    wallclock_budget_hours: float = Field(gt=0, default=8.0)


class GPUProfileConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = Field(min_length=1)
    dtype: _DTYPE
    vram_gb: int = Field(gt=0)
    recommended_batch_size: int = Field(gt=0)


class WandBConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    project: str = "vn-receipt-ocr"
    run_name_template: str = "{model_short}-r{lora_rank}-{date}-{nnn}"
    enabled: bool = True
    mode_fallback: Literal["offline", "disabled"] = "offline"


class HFHubConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    repo_owner: str
    repo_name_template: str = "vn-receipt-ocr-{run_id}"
    private: bool = True
    enabled: bool = True


class EvalConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    metrics_enabled: list[str] = Field(min_length=1)
    decode: _DECODE = "greedy"
    max_new_tokens: int = Field(gt=0, default=244)
    sample_table_size: int = 10


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: ModelConfig
    lora: LoRAConfig
    data: DataConfig
    trainer: TrainerConfig
    gpu_profile: GPUProfileConfig
    wandb: WandBConfig
    hf_hub: HFHubConfig
    eval: EvalConfig

    @field_validator("eval")
    @classmethod
    def _eval_metrics_known(cls, v: EvalConfig) -> EvalConfig:
        known = {"cer", "diacritic_cer", "cer_normalized", "wer", "edit_ops",
                 "length_ratio", "empty_pred_rate", "latency"}
        unknown = set(v.metrics_enabled) - known
        if unknown:
            raise ValueError(f"unknown metrics: {sorted(unknown)}")
        return v
