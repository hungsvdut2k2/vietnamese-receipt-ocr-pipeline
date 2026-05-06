import pytest
import torch

from vn_receipt_ocr.config.models import GPUProfileConfig, ModelConfig
from vn_receipt_ocr.model.loader import _VISION_PATH, resolve_dtype


def test_resolve_dtype_uses_gpu_profile():
    dt = resolve_dtype(
        model=ModelConfig(model_id="x"),
        gpu_profile=GPUProfileConfig(name="t4_16gb", dtype="bf16",
                                     vram_gb=16, recommended_batch_size=1),
    )
    assert dt is torch.bfloat16


def test_resolve_dtype_override_wins():
    dt = resolve_dtype(
        model=ModelConfig(model_id="x", dtype_override="fp16"),
        gpu_profile=GPUProfileConfig(name="t4_16gb", dtype="bf16",
                                     vram_gb=16, recommended_batch_size=1),
    )
    assert dt is torch.float16


@pytest.mark.parametrize("name", [
    "model.visual.blocks.0.attn.q_proj",
    "base_model.model.model.vision_tower.encoder.layers.5.k_proj",
    "vision_model.embed_tokens.weight",
])
def test_vision_path_matches_known_branches(name: str):
    assert _VISION_PATH.search(name)


@pytest.mark.parametrize("name", [
    "model.layers.0.self_attn.q_proj",
    "base_model.model.language_model.layers.10.mlp.gate_proj",
    "lm_head.weight",
])
def test_vision_path_does_not_match_language_branch(name: str):
    assert not _VISION_PATH.search(name)
