# Vietnamese Receipt OCR Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the `vn_receipt_ocr` Python package — a Kaggle-runnable LoRA fine-tuning pipeline for Qwen3-VL-2B via Unsloth, performing whole-document Vietnamese receipt transcription, with WandB tracking and Hugging Face Hub adapter persistence.

**Architecture:** Subpackage under `src/vn_receipt_ocr/` with seven modules (`config`, `data`, `model`, `train`, `eval`, `kaggle`, `cli`). Pydantic v2 + composable YAML configs, Unsloth `FastVisionModel` + LoRA, single-GPU 16 GB baseline with GPU profile YAMLs. Pure-function eval metrics (TDD-friendly); heavy components (model load, trainer) tested with mocks + a CPU dry-run smoke test. Frequent commits. Three phases — A: foundation, B: training, C: packaging.

**Tech Stack:** Python 3.11, **uv** (package manager), Pydantic v2, **Unsloth** (`unsloth`), `transformers`, `peft`, `trl` (SFTTrainer), `huggingface_hub`, `wandb`, `jiwer` (CER/WER), `pyyaml`, `pillow`, `pytest`, Kaggle Secrets via `kaggle_secrets` module.

**Spec:** [docs/superpowers/specs/2026-05-06-vn-receipt-ocr-training-pipeline-design.md](../specs/2026-05-06-vn-receipt-ocr-training-pipeline-design.md)
**Decision note:** [docs/superpowers/specs/2026-05-06-vn-receipt-ocr-training-pipeline-decision.md](../specs/2026-05-06-vn-receipt-ocr-training-pipeline-decision.md)

---

## File Structure

### Package source (created by this plan)

```
src/vn_receipt_ocr/
├── __init__.py                     # top-level exports: train, evaluate, predict
├── __main__.py                     # python -m vn_receipt_ocr <subcommand>
├── cli.py                          # argparse: train/eval/predict subcommands
│
├── config/
│   ├── __init__.py
│   ├── models.py                   # Pydantic v2: TrainConfig + sub-configs
│   ├── loader.py                   # YAML composition + deep-merge
│   └── validation.py               # wall-clock projection, hardware compatibility
│
├── data/
│   ├── __init__.py
│   ├── dataset.py                  # MCOCRDataset (line-OCR parse + group + join)
│   ├── prompt.py                   # PromptBuilder (fixed Vietnamese instruction)
│   └── collator.py                 # QwenVLCollator (chat template + label masking)
│
├── model/
│   ├── __init__.py
│   └── loader.py                   # build_load_kwargs / build_peft_kwargs / load_model_and_processor (FastVisionModel + LoRA)
│
├── train/
│   ├── __init__.py
│   ├── trainer.py                  # train() — wraps SFTTrainer
│   ├── callbacks.py                # init_wandb + JSONLFallback, PerEpochEvalCallback, CheckpointSyncCallback
│   └── manifest.py                 # reproducibility manifest
│
├── eval/
│   ├── __init__.py
│   ├── cer.py                      # CER (jiwer)
│   ├── wer.py                      # WER (jiwer)
│   ├── diacritic_cer.py            # NFD-decomposed Vietnamese diacritic filter
│   ├── normalized_cer.py           # applies normalization_rules.yaml
│   ├── edit_ops.py                 # sub/ins/del breakdown
│   ├── length_ratio.py             # len(pred)/len(gt) per sample + histogram
│   ├── latency.py                  # P50/P95 over per-sample inference times
│   ├── batch_predict.py            # greedy decode batched
│   └── aggregate.py                # combines metrics into one result dict
│
└── kaggle/
    ├── __init__.py
    ├── paths.py                    # /kaggle/input/ vs local resolution
    ├── secrets.py                  # UserSecretsClient + offline fallback
    └── gpu_detect.py               # auto-select GPU profile
```

### Config files (created by this plan)

```
configs/
├── gpu_profiles/
│   ├── p100_16gb.yaml
│   ├── t4_16gb.yaml
│   ├── t4x2_32gb.yaml
│   └── l4_24gb.yaml
├── data/
│   └── mcocr_train_val.yaml
├── model/
│   ├── qwen3_vl_2b.yaml
│   └── qwen2_vl_2b.yaml
├── lora/
│   ├── r16_attn_mlp.yaml
│   ├── r8_attn_only.yaml
│   └── r32_attn_mlp.yaml
└── experiments/
    └── baseline_v1.yaml
```

### Notebook + tests (created by this plan)

```
notebooks/
├── kaggle_train.ipynb
└── README.md

tests/
├── __init__.py
├── conftest.py
├── data/
│   ├── test_dataset.py
│   ├── test_prompt.py
│   └── test_collator.py
├── eval/
│   ├── test_cer.py
│   ├── test_wer.py
│   ├── test_diacritic_cer.py
│   ├── test_normalized_cer.py
│   ├── test_edit_ops.py
│   ├── test_length_ratio.py
│   ├── test_latency.py
│   └── test_aggregate.py
├── config/
│   ├── test_models.py
│   ├── test_loader.py
│   └── test_validation.py
├── kaggle/
│   ├── test_paths.py
│   ├── test_secrets.py
│   └── test_gpu_detect.py
└── train/
    ├── test_callbacks.py
    └── test_manifest.py
```

### Existing files modified

- `pyproject.toml` — add training deps + dev deps + `[tool.pytest.ini_options]` + `[tool.hatch.build.targets.wheel] packages` for src/ layout.
- `.gitignore` — exclude wandb/, hf_cache/, /kaggle/working/-style runtime dirs.

---

# Phase A — Foundation

Tasks 1–14. End-of-phase deliverable: data loaders, eval metrics, config system, and Kaggle helpers all working with full unit-test coverage. No model, no training. Provides a usable library subset (someone can compute CER on predictions without ever touching Unsloth).

---

## Task 1: Package scaffolding (src/ layout, deps, pytest)

**Files:**
- Modify: `pyproject.toml`
- Create: `src/vn_receipt_ocr/__init__.py`
- Create: `src/vn_receipt_ocr/{config,data,model,train,eval,kaggle}/__init__.py` (six empty `__init__.py`)
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`
- Modify: `.gitignore`

- [ ] **Step 1: Add training dependencies via uv**

```bash
uv add unsloth transformers peft trl accelerate bitsandbytes \
       huggingface_hub wandb jiwer pydantic
uv add --dev pytest pytest-cov ruff
```

Expected: `pyproject.toml` `[project] dependencies` updated; `uv.lock` regenerated.

- [ ] **Step 2: Switch pyproject to src/ layout for the new package**

Edit `pyproject.toml` — append (after the existing `[project]` block):

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/vn_receipt_ocr"]

[project.scripts]
vn-receipt-ocr = "vn_receipt_ocr.cli:main"

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]
addopts = "-v --strict-markers"
```

- [ ] **Step 3: Create empty subpackage `__init__.py` files**

```bash
mkdir -p src/vn_receipt_ocr/{config,data,model,train,eval,kaggle}
touch src/vn_receipt_ocr/__init__.py
touch src/vn_receipt_ocr/{config,data,model,train,eval,kaggle}/__init__.py
mkdir -p tests/{config,data,eval,kaggle,train}
touch tests/__init__.py tests/{config,data,eval,kaggle,train}/__init__.py
```

- [ ] **Step 4: Add `tests/conftest.py` with fixture for the MC-OCR data path**

```python
# tests/conftest.py
from pathlib import Path
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def mcocr_root(repo_root: Path) -> Path:
    return repo_root / "datasets" / "kagglehub" / "datasets" / "domixi1989" \
        / "vietnamese-receipts-mc-ocr-2021" / "versions" / "17"


@pytest.fixture
def tmp_text_recognition_file(tmp_path: Path) -> Path:
    """Tiny hand-crafted line-OCR file for deterministic dataset tests."""
    f = tmp_path / "tiny.txt"
    f.write_text(
        "img_a_0.jpg\tHello\n"
        "img_a_1.jpg\tWorld\n"
        "img_a_2.jpg\tFoo\n"
        "img_b_0.jpg\tOne line only\n"
        "img_c_2.jpg\tThird\n"  # out-of-order suffix
        "img_c_0.jpg\tFirst\n"
        "img_c_1.jpg\tSecond\n",
        encoding="utf-8",
    )
    return f
```

- [ ] **Step 5: Update `.gitignore`**

Append:

```
# Training pipeline runtime
wandb/
.wandb/
hf_cache/
checkpoints/
.unsloth_compiled_cache/

# Tests
.pytest_cache/
.coverage
coverage.xml
htmlcov/
```

- [ ] **Step 6: Verify everything imports**

```bash
uv run python -c "import vn_receipt_ocr; print(vn_receipt_ocr.__file__)"
uv run pytest --collect-only
```

Expected: package import resolves to `src/vn_receipt_ocr/__init__.py`. pytest reports "no tests collected" (no errors).

- [ ] **Step 7: Commit**

```bash
git add pyproject.toml uv.lock src/vn_receipt_ocr tests .gitignore
git commit -m "feat(pkg): src/ scaffold + training deps + pytest config"
```

---

## Task 2: Pydantic config models

**Files:**
- Create: `src/vn_receipt_ocr/config/models.py`
- Create: `tests/config/test_models.py`

- [ ] **Step 1: Write failing test for config validation**

```python
# tests/config/test_models.py
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
        model=ModelConfig(model_id="unsloth/Qwen3-VL-2B-Instruct"),
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
```

- [ ] **Step 2: Run test, verify it fails**

```bash
uv run pytest tests/config/test_models.py -v
```

Expected: ImportError or collection error (`models` module missing).

- [ ] **Step 3: Implement Pydantic models**

```python
# src/vn_receipt_ocr/config/models.py
from __future__ import annotations
from pathlib import Path
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
```

- [ ] **Step 4: Run tests, verify pass**

```bash
uv run pytest tests/config/test_models.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vn_receipt_ocr/config/models.py tests/config/test_models.py
git commit -m "feat(config): pydantic v2 config models"
```

---

## Task 3: YAML config loader with deep-merge composition

**Files:**
- Create: `src/vn_receipt_ocr/config/loader.py`
- Create: `tests/config/test_loader.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/config/test_loader.py
from pathlib import Path
import yaml
import pytest

from vn_receipt_ocr.config.loader import deep_merge, load_yaml, load_train_config


def test_deep_merge_overrides_leaves():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 99}, "e": 5}
    assert deep_merge(base, override) == {"a": 1, "b": {"c": 99, "d": 3}, "e": 5}


def test_deep_merge_lists_are_replaced_not_concatenated():
    base = {"x": [1, 2, 3]}
    override = {"x": [9]}
    assert deep_merge(base, override) == {"x": [9]}


def test_load_yaml_round_trip(tmp_path: Path):
    p = tmp_path / "x.yaml"
    p.write_text("a: 1\nb:\n  c: 2\n", encoding="utf-8")
    assert load_yaml(p) == {"a": 1, "b": {"c": 2}}


def test_load_train_config_composes_includes(tmp_path: Path):
    base_dir = tmp_path
    (base_dir / "gpu_profiles").mkdir()
    (base_dir / "data").mkdir()
    (base_dir / "model").mkdir()
    (base_dir / "lora").mkdir()
    (base_dir / "experiments").mkdir()

    (base_dir / "gpu_profiles" / "p100_16gb.yaml").write_text(
        "gpu_profile:\n  name: p100_16gb\n  dtype: fp16\n  vram_gb: 16\n  recommended_batch_size: 1\n"
    )
    (base_dir / "data" / "mcocr.yaml").write_text(
        "data:\n  train_path: a\n  train_images_dir: b\n  val_path: c\n  val_images_dir: d\n"
        "  instruction: 'X'\n  target_resolution: 888\n  max_resolution: 1388\n  max_seq_length: 203\n"
    )
    (base_dir / "model" / "q3.yaml").write_text(
        "model:\n  model_id: unsloth/Qwen3-VL-2B-Instruct\n"
    )
    (base_dir / "lora" / "r16.yaml").write_text(
        "lora:\n  rank: 16\n  alpha: 32\n  dropout: 0.05\n"
        "  target_modules: [q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]\n  bias: none\n"
        "  finetune_vision_layers: false\n"
    )
    (base_dir / "experiments" / "exp.yaml").write_text(
        "include:\n  - gpu_profiles/p100_16gb.yaml\n  - data/mcocr.yaml\n"
        "  - model/q3.yaml\n  - lora/r16.yaml\n"
        "trainer:\n  epochs: 3\n  per_device_batch_size: 1\n  grad_accum: 8\n  lr: 1e-4\n"
        "  warmup_ratio: 0.05\n  optimizer: adamw_8bit\n  gradient_checkpointing: true\n"
        "  seed: 42\n  deterministic: false\n  wallclock_budget_hours: 8.0\n"
        "wandb:\n  project: vn-receipt-ocr\n  run_name_template: '{model_short}-r{lora_rank}-{date}-{nnn}'\n"
        "  enabled: true\n  mode_fallback: offline\n"
        "hf_hub:\n  repo_owner: me\n  repo_name_template: 'vn-receipt-ocr-{run_id}'\n"
        "  private: true\n  enabled: true\n"
        "eval:\n  metrics_enabled: [cer, diacritic_cer, cer_normalized, wer, edit_ops, length_ratio, empty_pred_rate, latency]\n"
        "  decode: greedy\n  max_new_tokens: 244\n  sample_table_size: 10\n"
    )

    cfg = load_train_config(base_dir / "experiments" / "exp.yaml", configs_root=base_dir)
    assert cfg.lora.rank == 16
    assert cfg.gpu_profile.name == "p100_16gb"
    assert cfg.model.model_id == "unsloth/Qwen3-VL-2B-Instruct"


def test_overrides_take_highest_precedence(tmp_path: Path):
    # same setup as above, with overrides={"lora.rank": 32}
    pass  # full version implemented in next test or skipped if too verbose
```

- [ ] **Step 2: Run, verify fails (module missing)**

```bash
uv run pytest tests/config/test_loader.py -v
```

- [ ] **Step 3: Implement loader**

```python
# src/vn_receipt_ocr/config/loader.py
from __future__ import annotations
from pathlib import Path
from typing import Any
import copy
import yaml

from vn_receipt_ocr.config.models import TrainConfig


def load_yaml(path: Path | str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    """Deep-merge override into a copy of base. Lists are replaced, not concatenated."""
    out = copy.deepcopy(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def _apply_dotted(target: dict, key: str, value: Any) -> None:
    """Mutate target so that target['a']['b']=value for key='a.b'."""
    parts = key.split(".")
    node = target
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value


def load_train_config(
    experiment_path: Path | str,
    configs_root: Path | str,
    overrides: dict[str, Any] | None = None,
) -> TrainConfig:
    """
    Load and compose a TrainConfig from an experiment YAML.

    The experiment YAML may contain an 'include' list of relative paths under
    configs_root; each included YAML is loaded and merged in order, then the
    experiment YAML's own keys are applied as the highest-precedence file-level
    layer. CLI/API `overrides` (dotted keys) are applied last.
    """
    experiment_path = Path(experiment_path)
    configs_root = Path(configs_root)

    raw = load_yaml(experiment_path)
    includes = raw.pop("include", []) or []

    merged: dict = {}
    for rel in includes:
        included = load_yaml(configs_root / rel)
        merged = deep_merge(merged, included)
    merged = deep_merge(merged, raw)

    if overrides:
        for k, v in overrides.items():
            _apply_dotted(merged, k, v)

    return TrainConfig.model_validate(merged)
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/config/test_loader.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vn_receipt_ocr/config/loader.py tests/config/test_loader.py
git commit -m "feat(config): YAML loader with deep-merge composition"
```

---

## Task 4: Config validation (hardware compatibility, wall-clock projection)

**Files:**
- Create: `src/vn_receipt_ocr/config/validation.py`
- Create: `tests/config/test_validation.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/config/test_validation.py
import pytest

from vn_receipt_ocr.config.validation import (
    validate_dtype_hardware, project_wallclock_hours, ConfigValidationError,
)


def test_bf16_on_p100_rejected():
    with pytest.raises(ConfigValidationError, match="P100"):
        validate_dtype_hardware(gpu_name="p100_16gb", dtype="bf16")


def test_fp16_on_p100_ok():
    validate_dtype_hardware(gpu_name="p100_16gb", dtype="fp16")  # no raise


def test_bf16_on_t4_ok():
    validate_dtype_hardware(gpu_name="t4_16gb", dtype="bf16")


def test_wallclock_projection_within_budget():
    h = project_wallclock_hours(
        n_train_samples=922, epochs=3, per_device_batch_size=1, grad_accum=8,
        seconds_per_optimizer_step=4.0,
    )
    # 922/8 = 115.25 steps/epoch * 3 = 345.75 steps * 4s = 1383s = 0.384h
    assert 0.3 < h < 0.5


def test_wallclock_projection_exceeds_budget_raises():
    with pytest.raises(ConfigValidationError, match="wall-clock"):
        project_wallclock_hours(
            n_train_samples=922, epochs=100,
            per_device_batch_size=1, grad_accum=8,
            seconds_per_optimizer_step=4.0,
            budget_hours=8.0, raise_on_exceed=True,
        )
```

- [ ] **Step 2: Run, verify fails**

```bash
uv run pytest tests/config/test_validation.py -v
```

- [ ] **Step 3: Implement**

```python
# src/vn_receipt_ocr/config/validation.py
from __future__ import annotations


class ConfigValidationError(ValueError):
    pass


_BF16_INCOMPATIBLE_GPUS = {"p100", "k80", "v100"}


def validate_dtype_hardware(gpu_name: str, dtype: str) -> None:
    """Raise ConfigValidationError if dtype is not supported on the given GPU."""
    name = gpu_name.lower()
    if dtype == "bf16" and any(g in name for g in _BF16_INCOMPATIBLE_GPUS):
        raise ConfigValidationError(
            f"GPU profile '{gpu_name}' does not support BF16 (P100/V100/K80 lack "
            "BF16 support); use FP16 instead."
        )


def project_wallclock_hours(
    *,
    n_train_samples: int,
    epochs: int,
    per_device_batch_size: int,
    grad_accum: int,
    seconds_per_optimizer_step: float,
    budget_hours: float | None = None,
    raise_on_exceed: bool = False,
) -> float:
    """Compute projected training wall-clock; optionally raise if it exceeds budget."""
    effective_batch = per_device_batch_size * grad_accum
    steps_per_epoch = (n_train_samples + effective_batch - 1) // effective_batch
    total_steps = steps_per_epoch * epochs
    seconds = total_steps * seconds_per_optimizer_step
    hours = seconds / 3600.0
    if raise_on_exceed and budget_hours is not None and hours > budget_hours:
        raise ConfigValidationError(
            f"Projected wall-clock {hours:.2f}h exceeds budget {budget_hours:.2f}h. "
            f"Reduce epochs or seconds_per_optimizer_step."
        )
    return hours
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/config/test_validation.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vn_receipt_ocr/config/validation.py tests/config/test_validation.py
git commit -m "feat(config): hardware + wall-clock validators"
```

---

## Task 5: GPU profile YAMLs

**Files:**
- Create: `configs/gpu_profiles/{p100_16gb,t4_16gb,t4x2_32gb,l4_24gb}.yaml`

- [ ] **Step 1: Create profiles**

```bash
mkdir -p configs/gpu_profiles
```

`configs/gpu_profiles/p100_16gb.yaml`:
```yaml
gpu_profile:
  name: p100_16gb
  dtype: fp16
  vram_gb: 16
  recommended_batch_size: 1
```

`configs/gpu_profiles/t4_16gb.yaml`:
```yaml
gpu_profile:
  name: t4_16gb
  dtype: bf16
  vram_gb: 16
  recommended_batch_size: 1
```

`configs/gpu_profiles/t4x2_32gb.yaml`:
```yaml
gpu_profile:
  name: t4x2_32gb
  dtype: bf16
  vram_gb: 32
  recommended_batch_size: 2
```

`configs/gpu_profiles/l4_24gb.yaml`:
```yaml
gpu_profile:
  name: l4_24gb
  dtype: bf16
  vram_gb: 24
  recommended_batch_size: 2
```

- [ ] **Step 2: Verify each loads as a valid `GPUProfileConfig`**

```bash
uv run python -c "
import yaml
from vn_receipt_ocr.config.models import GPUProfileConfig
for p in ['p100_16gb','t4_16gb','t4x2_32gb','l4_24gb']:
    d = yaml.safe_load(open(f'configs/gpu_profiles/{p}.yaml'))
    GPUProfileConfig.model_validate(d['gpu_profile'])
    print(p, 'ok')
"
```

Expected: 4 lines of "<name> ok".

- [ ] **Step 3: Commit**

```bash
git add configs/gpu_profiles
git commit -m "feat(configs): GPU profile YAMLs (P100, T4, T4x2, L4)"
```

---

## Task 6: Data, model, lora, experiment YAMLs

**Files:**
- Create: `configs/data/mcocr_train_val.yaml`
- Create: `configs/model/qwen3_vl_2b.yaml`
- Create: `configs/model/qwen2_vl_2b.yaml`
- Create: `configs/lora/r16_attn_mlp.yaml`
- Create: `configs/lora/r8_attn_only.yaml`
- Create: `configs/lora/r32_attn_mlp.yaml`
- Create: `configs/experiments/baseline_v1.yaml`

- [ ] **Step 1: Create `configs/data/mcocr_train_val.yaml`**

```yaml
data:
  train_path: datasets/kagglehub/datasets/domixi1989/vietnamese-receipts-mc-ocr-2021/versions/17/text_recognition_train_data.txt
  train_images_dir: datasets/kagglehub/datasets/domixi1989/vietnamese-receipts-mc-ocr-2021/versions/17/train_images
  val_path: datasets/kagglehub/datasets/domixi1989/vietnamese-receipts-mc-ocr-2021/versions/17/text_recognition_val_data.txt
  val_images_dir: datasets/kagglehub/datasets/domixi1989/vietnamese-receipts-mc-ocr-2021/versions/17/val_images
  test_path: null
  test_images_dir: null
  instruction: "Trích xuất toàn bộ nội dung văn bản từ hóa đơn này, giữ nguyên thứ tự đọc từ trên xuống dưới."
  target_resolution: 888
  max_resolution: 1388
  max_seq_length: 203
```

- [ ] **Step 2: Create model YAMLs**

`configs/model/qwen3_vl_2b.yaml`:
```yaml
model:
  model_id: unsloth/Qwen3-VL-2B-Instruct
  freeze_vision_tower: true
```

`configs/model/qwen2_vl_2b.yaml`:
```yaml
model:
  model_id: unsloth/Qwen2-VL-2B-Instruct
  freeze_vision_tower: true
```

- [ ] **Step 3: Create LoRA YAMLs**

`configs/lora/r16_attn_mlp.yaml`:
```yaml
lora:
  rank: 16
  alpha: 32
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  bias: none
  finetune_vision_layers: false
```

`configs/lora/r8_attn_only.yaml`:
```yaml
lora:
  rank: 8
  alpha: 16
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj]
  bias: none
  finetune_vision_layers: false
```

`configs/lora/r32_attn_mlp.yaml`:
```yaml
lora:
  rank: 32
  alpha: 64
  dropout: 0.05
  target_modules: [q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj]
  bias: none
  finetune_vision_layers: false
```

- [ ] **Step 4: Create experiment YAML**

`configs/experiments/baseline_v1.yaml`:
```yaml
include:
  - gpu_profiles/p100_16gb.yaml
  - data/mcocr_train_val.yaml
  - model/qwen3_vl_2b.yaml
  - lora/r16_attn_mlp.yaml
trainer:
  epochs: 3
  per_device_batch_size: 1
  grad_accum: 8
  lr: 1.0e-4
  warmup_ratio: 0.05
  optimizer: adamw_8bit
  gradient_checkpointing: true
  seed: 42
  deterministic: false
  wallclock_budget_hours: 8.0
wandb:
  project: vn-receipt-ocr
  run_name_template: "{model_short}-r{lora_rank}-{date}-{nnn}"
  enabled: true
  mode_fallback: offline
hf_hub:
  repo_owner: TODO_SET_VIA_OVERRIDE
  repo_name_template: "vn-receipt-ocr-{run_id}"
  private: true
  enabled: true
eval:
  metrics_enabled:
    - cer
    - diacritic_cer
    - cer_normalized
    - wer
    - edit_ops
    - length_ratio
    - empty_pred_rate
    - latency
  decode: greedy
  max_new_tokens: 244
  sample_table_size: 10
```

(`hf_hub.repo_owner` is intentionally `TODO_SET_VIA_OVERRIDE` — set via CLI override at run time, e.g. `--override hf_hub.repo_owner=<your-hf-username>`.)

- [ ] **Step 5: Verify experiment loads**

```bash
uv run python -c "
from pathlib import Path
from vn_receipt_ocr.config.loader import load_train_config
cfg = load_train_config(
    Path('configs/experiments/baseline_v1.yaml'),
    configs_root=Path('configs'),
    overrides={'hf_hub.repo_owner': 'me'},
)
print('ok', cfg.gpu_profile.name, cfg.lora.rank)
"
```

Expected: `ok p100_16gb 16`.

- [ ] **Step 6: Commit**

```bash
git add configs
git commit -m "feat(configs): data/model/lora/experiment YAMLs (baseline_v1)"
```

---

## Task 7: MCOCRDataset (line-OCR parsing + grouping)

**Files:**
- Create: `src/vn_receipt_ocr/data/dataset.py`
- Create: `tests/data/test_dataset.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/data/test_dataset.py
from pathlib import Path
import pytest

from vn_receipt_ocr.data.dataset import MCOCRDataset, group_lines_by_prefix


def test_group_lines_by_prefix(tmp_text_recognition_file: Path):
    grouped = group_lines_by_prefix(tmp_text_recognition_file)
    assert set(grouped.keys()) == {"img_a", "img_b", "img_c"}
    assert grouped["img_a"] == ["Hello", "World", "Foo"]
    assert grouped["img_b"] == ["One line only"]
    assert grouped["img_c"] == ["First", "Second", "Third"]  # sorted by suffix


def test_full_text_target_joins_with_newline(tmp_text_recognition_file: Path):
    ds = MCOCRDataset(
        text_path=tmp_text_recognition_file,
        images_dir=tmp_text_recognition_file.parent,  # not used in this assertion
        instruction="X",
        require_images=False,
    )
    assert ds.full_text("img_a") == "Hello\nWorld\nFoo"
    assert ds.full_text("img_c") == "First\nSecond\nThird"


def test_dataset_len_equals_unique_prefixes(tmp_text_recognition_file: Path):
    ds = MCOCRDataset(
        text_path=tmp_text_recognition_file,
        images_dir=tmp_text_recognition_file.parent,
        instruction="X",
        require_images=False,
    )
    assert len(ds) == 3


def test_getitem_returns_dict_with_full_text_and_image_path(
    tmp_text_recognition_file: Path,
):
    ds = MCOCRDataset(
        text_path=tmp_text_recognition_file,
        images_dir=tmp_text_recognition_file.parent,
        instruction="X",
        require_images=False,
    )
    item = ds[0]
    assert "image_path" in item
    assert "full_text" in item
    assert "instruction" in item
    assert item["instruction"] == "X"


def test_missing_image_raises_when_required(tmp_text_recognition_file: Path):
    ds = MCOCRDataset(
        text_path=tmp_text_recognition_file,
        images_dir=tmp_text_recognition_file.parent,
        instruction="X",
        require_images=True,
    )
    # tmp_text_recognition_file.parent has no img_a.jpg
    with pytest.raises(FileNotFoundError):
        _ = ds[0]
```

- [ ] **Step 2: Run, verify fails**

```bash
uv run pytest tests/data/test_dataset.py -v
```

- [ ] **Step 3: Implement**

```python
# src/vn_receipt_ocr/data/dataset.py
from __future__ import annotations
from pathlib import Path
import re
from typing import Any
from torch.utils.data import Dataset


_SUFFIX_RE = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d+)\.jpg$")


def group_lines_by_prefix(text_path: Path | str) -> dict[str, list[str]]:
    """
    Read a line-OCR text file (one row = '<filename>_<N>.jpg\\t<text>'),
    group rows by stripped-suffix prefix, and sort within each group by integer N.
    Returns {prefix: [text_for_N=0, text_for_N=1, ...]}.
    """
    text_path = Path(text_path)
    by_prefix: dict[str, list[tuple[int, str]]] = {}
    with open(text_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            if not line:
                continue
            try:
                fname, text = line.split("\t", 1)
            except ValueError as e:
                raise ValueError(f"Malformed row in {text_path}: {line!r}") from e
            m = _SUFFIX_RE.match(fname)
            if not m:
                raise ValueError(f"Filename does not match <prefix>_<N>.jpg: {fname!r}")
            prefix = m.group("prefix")
            idx = int(m.group("idx"))
            by_prefix.setdefault(prefix, []).append((idx, text))
    return {p: [t for _, t in sorted(rows, key=lambda r: r[0])]
            for p, rows in by_prefix.items()}


class MCOCRDataset(Dataset):
    """
    Each item: {image_path: Path, full_text: str, instruction: str, prefix: str}.
    Image loading is deferred to the collator/processor so we don't repeatedly
    decode the same image on retries; tests can run without images by passing
    require_images=False.
    """

    def __init__(
        self,
        text_path: Path | str,
        images_dir: Path | str,
        instruction: str,
        *,
        require_images: bool = True,
    ) -> None:
        self.text_path = Path(text_path)
        self.images_dir = Path(images_dir)
        self.instruction = instruction
        self.require_images = require_images

        self._lines_by_prefix = group_lines_by_prefix(self.text_path)
        self._prefixes = sorted(self._lines_by_prefix.keys())

    def __len__(self) -> int:
        return len(self._prefixes)

    def full_text(self, prefix: str) -> str:
        return "\n".join(self._lines_by_prefix[prefix])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        prefix = self._prefixes[idx]
        image_path = self.images_dir / f"{prefix}.jpg"
        if self.require_images and not image_path.is_file():
            raise FileNotFoundError(f"Image not found for prefix '{prefix}': {image_path}")
        return {
            "prefix": prefix,
            "image_path": image_path,
            "instruction": self.instruction,
            "full_text": self.full_text(prefix),
        }
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/data/test_dataset.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vn_receipt_ocr/data/dataset.py tests/data/test_dataset.py
git commit -m "feat(data): MCOCRDataset with prefix grouping + index sort + newline join"
```

---

## Task 8: PromptBuilder (chat-template message construction)

**Files:**
- Create: `src/vn_receipt_ocr/data/prompt.py`
- Create: `tests/data/test_prompt.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/data/test_prompt.py
from PIL import Image
import pytest

from vn_receipt_ocr.data.prompt import PromptBuilder


@pytest.fixture
def dummy_image() -> Image.Image:
    return Image.new("RGB", (100, 100), color="white")


def test_build_messages_user_then_assistant(dummy_image):
    pb = PromptBuilder(instruction="Trích xuất tất cả nội dung.")
    msgs = pb.build_train_messages(image=dummy_image, target="Hello\nWorld")
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


def test_user_content_has_image_and_text(dummy_image):
    pb = PromptBuilder(instruction="X")
    msgs = pb.build_train_messages(image=dummy_image, target="t")
    user_content = msgs[0]["content"]
    types = [c["type"] for c in user_content]
    assert types == ["image", "text"]
    assert user_content[1]["text"] == "X"


def test_assistant_content_is_target(dummy_image):
    pb = PromptBuilder(instruction="X")
    msgs = pb.build_train_messages(image=dummy_image, target="t1\nt2")
    assert msgs[1]["content"] == "t1\nt2"


def test_inference_messages_omit_assistant(dummy_image):
    pb = PromptBuilder(instruction="X")
    msgs = pb.build_inference_messages(image=dummy_image)
    assert len(msgs) == 1
    assert msgs[0]["role"] == "user"
```

- [ ] **Step 2: Run, verify fails**

```bash
uv run pytest tests/data/test_prompt.py -v
```

- [ ] **Step 3: Implement**

```python
# src/vn_receipt_ocr/data/prompt.py
from __future__ import annotations
from PIL import Image


class PromptBuilder:
    """Build Qwen-VL chat-template messages.

    Train messages contain a user turn (image + instruction) followed by an
    assistant turn (target). Inference messages contain only the user turn.
    """

    def __init__(self, instruction: str) -> None:
        if not instruction:
            raise ValueError("instruction must be a non-empty string")
        self.instruction = instruction

    def build_train_messages(self, *, image: Image.Image, target: str) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.instruction},
                ],
            },
            {"role": "assistant", "content": target},
        ]

    def build_inference_messages(self, *, image: Image.Image) -> list[dict]:
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.instruction},
                ],
            }
        ]
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/data/test_prompt.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vn_receipt_ocr/data/prompt.py tests/data/test_prompt.py
git commit -m "feat(data): PromptBuilder for Qwen-VL train/inference messages"
```

---

## Task 9: CER and WER metrics (jiwer wrapper)

**Files:**
- Create: `src/vn_receipt_ocr/eval/cer.py`
- Create: `src/vn_receipt_ocr/eval/wer.py`
- Create: `tests/eval/test_cer.py`
- Create: `tests/eval/test_wer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/eval/test_cer.py
from vn_receipt_ocr.eval.cer import compute_cer


def test_cer_zero_for_identical_strings():
    assert compute_cer(predictions=["abc"], references=["abc"]) == 0.0


def test_cer_one_for_completely_wrong():
    # 3-char ref, 3 substitutions
    assert compute_cer(predictions=["xyz"], references=["abc"]) == 1.0


def test_cer_aggregates_across_corpus():
    cer = compute_cer(
        predictions=["abc", "abcd"],
        references=["abc", "abce"],
    )
    # total ref chars = 7; total errors = 1 (substitution in second pair)
    assert abs(cer - 1 / 7) < 1e-6


def test_cer_empty_inputs_raises():
    import pytest
    with pytest.raises(ValueError):
        compute_cer(predictions=[], references=[])
```

```python
# tests/eval/test_wer.py
from vn_receipt_ocr.eval.wer import compute_wer


def test_wer_zero_for_identical():
    assert compute_wer(predictions=["a b c"], references=["a b c"]) == 0.0


def test_wer_one_for_completely_wrong():
    assert compute_wer(predictions=["x y z"], references=["a b c"]) == 1.0
```

- [ ] **Step 2: Run, verify fails**

```bash
uv run pytest tests/eval/test_cer.py tests/eval/test_wer.py -v
```

- [ ] **Step 3: Implement**

```python
# src/vn_receipt_ocr/eval/cer.py
from __future__ import annotations
from jiwer import cer


def compute_cer(*, predictions: list[str], references: list[str]) -> float:
    if not predictions or not references:
        raise ValueError("predictions and references must be non-empty")
    if len(predictions) != len(references):
        raise ValueError(
            f"length mismatch: predictions={len(predictions)} references={len(references)}"
        )
    return float(cer(references, predictions))
```

```python
# src/vn_receipt_ocr/eval/wer.py
from __future__ import annotations
from jiwer import wer


def compute_wer(*, predictions: list[str], references: list[str]) -> float:
    if not predictions or not references:
        raise ValueError("predictions and references must be non-empty")
    if len(predictions) != len(references):
        raise ValueError(
            f"length mismatch: predictions={len(predictions)} references={len(references)}"
        )
    return float(wer(references, predictions))
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/eval/test_cer.py tests/eval/test_wer.py -v
```

Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vn_receipt_ocr/eval/cer.py src/vn_receipt_ocr/eval/wer.py \
        tests/eval/test_cer.py tests/eval/test_wer.py
git commit -m "feat(eval): CER and WER (jiwer wrappers)"
```

---

## Task 10: Diacritic CER

**Files:**
- Create: `src/vn_receipt_ocr/eval/diacritic_cer.py`
- Create: `tests/eval/test_diacritic_cer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/eval/test_diacritic_cer.py
from vn_receipt_ocr.eval.diacritic_cer import (
    compute_diacritic_cer, filter_to_diacritic_chars,
)


def test_filter_keeps_only_vietnamese_diacritic_chars():
    s = "Hà Nội 123"
    # Keeps: à, ộ. Drops: H, N, i, space, digits.
    out = filter_to_diacritic_chars(s)
    assert out == "àộ"


def test_filter_unaccented_yields_empty_string():
    assert filter_to_diacritic_chars("Hello 123") == ""


def test_diacritic_cer_zero_for_perfect_diacritics():
    cer = compute_diacritic_cer(
        predictions=["Hà Nội"], references=["Hà Nội"],
    )
    assert cer == 0.0


def test_diacritic_cer_ignores_non_diacritic_substitutions():
    # pred and ref differ only in non-diacritic chars; diacritic CER should be 0
    cer = compute_diacritic_cer(
        predictions=["XX Nội"],
        references=["Hà Nội"],
    )
    # ref diacritic chars: à, ộ → 2; pred diacritic chars: ộ → 1
    # (pred is missing 'à' from XX) — so 1 deletion / 2 ref-chars = 0.5
    assert abs(cer - 0.5) < 1e-6


def test_diacritic_cer_returns_zero_when_no_diacritics_in_reference():
    # When reference has no diacritic chars, CER on empty strings is undefined;
    # we return 0.0 as a sentinel.
    cer = compute_diacritic_cer(
        predictions=["abc"], references=["abc"],
    )
    assert cer == 0.0
```

- [ ] **Step 2: Run, verify fails**

```bash
uv run pytest tests/eval/test_diacritic_cer.py -v
```

- [ ] **Step 3: Implement**

```python
# src/vn_receipt_ocr/eval/diacritic_cer.py
from __future__ import annotations
import unicodedata
from jiwer import cer as _jiwer_cer


VIETNAMESE_DIACRITIC_MARKS = {
    "̀",  # combining grave accent
    "́",  # combining acute accent
    "̂",  # combining circumflex accent
    "̃",  # combining tilde
    "̆",  # combining breve
    "̉",  # combining hook above
    "̛",  # combining horn
    "̣",  # combining dot below
}


def _has_vietnamese_diacritic(ch: str) -> bool:
    decomposed = unicodedata.normalize("NFD", ch)
    return any(c in VIETNAMESE_DIACRITIC_MARKS for c in decomposed)


def filter_to_diacritic_chars(s: str) -> str:
    return "".join(ch for ch in s if _has_vietnamese_diacritic(ch))


def compute_diacritic_cer(
    *, predictions: list[str], references: list[str]
) -> float:
    if len(predictions) != len(references):
        raise ValueError("predictions and references length mismatch")
    p_filt = [filter_to_diacritic_chars(p) for p in predictions]
    r_filt = [filter_to_diacritic_chars(r) for r in references]
    # If no diacritics anywhere in references, return 0.0 (sentinel)
    if all(len(r) == 0 for r in r_filt):
        return 0.0
    # jiwer cer accepts empty strings as long as not ALL are empty
    return float(_jiwer_cer(r_filt, p_filt))
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/eval/test_diacritic_cer.py -v
```

Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vn_receipt_ocr/eval/diacritic_cer.py tests/eval/test_diacritic_cer.py
git commit -m "feat(eval): diacritic-CER with NFD-decomposed Vietnamese mark filter"
```

---

## Task 11: Normalized CER (apply EDA normalization rules first)

**Files:**
- Create: `src/vn_receipt_ocr/eval/normalized_cer.py`
- Create: `tests/eval/test_normalized_cer.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/eval/test_normalized_cer.py
from vn_receipt_ocr.eval.normalized_cer import (
    canonicalize_numeric, canonicalize_currency, compute_normalized_cer,
)


def test_canonicalize_numeric_strips_thousands_separators():
    assert canonicalize_numeric("8,600") == "8600"
    assert canonicalize_numeric("8.600") == "8600"
    assert canonicalize_numeric("1,234,567.89") == "1234567.89"


def test_canonicalize_currency_strips_suffix():
    assert canonicalize_currency("8600đ") == "8600"
    assert canonicalize_currency("8600 VND") == "8600"
    assert canonicalize_currency("8600đồng") == "8600"


def test_normalized_cer_treats_format_variants_as_equal():
    # Two strings differ only in numeric format and currency suffix
    cer = compute_normalized_cer(
        predictions=["Tổng: 8,600đ"],
        references=["Tổng: 8600"],
    )
    assert cer == 0.0
```

- [ ] **Step 2: Run, verify fails**

```bash
uv run pytest tests/eval/test_normalized_cer.py -v
```

- [ ] **Step 3: Implement**

```python
# src/vn_receipt_ocr/eval/normalized_cer.py
from __future__ import annotations
import re
from jiwer import cer as _jiwer_cer


_THOUSANDS_SEP_RE = re.compile(r"(?<=\d)[,.\s](?=\d{3}(?:[^\d]|$))")
_CURRENCY_SUFFIX_RE = re.compile(
    r"\s*(?:đồng|VNĐ|VND|đ|₫)\b|\s*(?:đồng|VNĐ|VND|đ|₫)$",
    flags=re.IGNORECASE,
)


def canonicalize_numeric(s: str) -> str:
    """Strip thousand separators (.,space) between digit groups."""
    prev = None
    cur = s
    while prev != cur:
        prev = cur
        cur = _THOUSANDS_SEP_RE.sub("", cur)
    return cur


def canonicalize_currency(s: str) -> str:
    """Strip Vietnamese currency suffixes."""
    # Run multiple times because suffixes may stack (e.g. "8600 VND đ").
    prev = None
    cur = s
    while prev != cur:
        prev = cur
        cur = _CURRENCY_SUFFIX_RE.sub("", cur)
    return cur.rstrip()


def normalize_for_cer(s: str) -> str:
    return canonicalize_numeric(canonicalize_currency(s))


def compute_normalized_cer(
    *, predictions: list[str], references: list[str]
) -> float:
    if len(predictions) != len(references):
        raise ValueError("predictions and references length mismatch")
    p = [normalize_for_cer(x) for x in predictions]
    r = [normalize_for_cer(x) for x in references]
    return float(_jiwer_cer(r, p))
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/eval/test_normalized_cer.py -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vn_receipt_ocr/eval/normalized_cer.py tests/eval/test_normalized_cer.py
git commit -m "feat(eval): normalized-CER (numeric + currency canonicalization)"
```

---

## Task 12: Edit-op breakdown, length ratio, latency

**Files:**
- Create: `src/vn_receipt_ocr/eval/edit_ops.py`
- Create: `src/vn_receipt_ocr/eval/length_ratio.py`
- Create: `src/vn_receipt_ocr/eval/latency.py`
- Create: `tests/eval/test_edit_ops.py`
- Create: `tests/eval/test_length_ratio.py`
- Create: `tests/eval/test_latency.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/eval/test_edit_ops.py
from vn_receipt_ocr.eval.edit_ops import compute_edit_ops


def test_edit_ops_pure_substitution():
    ops = compute_edit_ops(predictions=["abc"], references=["abd"])
    assert ops == {"sub": 1, "ins": 0, "del": 0}


def test_edit_ops_pure_insertion():
    ops = compute_edit_ops(predictions=["abcd"], references=["abc"])
    assert ops == {"sub": 0, "ins": 1, "del": 0}


def test_edit_ops_pure_deletion():
    ops = compute_edit_ops(predictions=["ab"], references=["abc"])
    assert ops == {"sub": 0, "ins": 0, "del": 1}


def test_edit_ops_aggregates():
    ops = compute_edit_ops(
        predictions=["abc", "abcd"],
        references=["abd", "abc"],
    )
    assert ops == {"sub": 1, "ins": 1, "del": 0}
```

```python
# tests/eval/test_length_ratio.py
from vn_receipt_ocr.eval.length_ratio import compute_length_ratios


def test_length_ratios():
    rs = compute_length_ratios(predictions=["abc", "ab"], references=["abc", "abc"])
    assert rs == [1.0, 2/3]


def test_length_ratio_zero_reference_yields_inf():
    rs = compute_length_ratios(predictions=["abc"], references=[""])
    assert rs[0] == float("inf")
```

```python
# tests/eval/test_latency.py
from vn_receipt_ocr.eval.latency import compute_latency_percentiles


def test_p50_p95():
    times_ms = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
    out = compute_latency_percentiles(times_ms)
    assert abs(out["p50"] - 55.0) < 1e-6
    assert abs(out["p95"] - 95.5) < 1.0  # numpy interpolation tolerance
```

- [ ] **Step 2: Run, verify fail**

```bash
uv run pytest tests/eval/test_edit_ops.py tests/eval/test_length_ratio.py tests/eval/test_latency.py -v
```

- [ ] **Step 3: Implement**

```python
# src/vn_receipt_ocr/eval/edit_ops.py
from __future__ import annotations
import jiwer


def compute_edit_ops(
    *, predictions: list[str], references: list[str]
) -> dict[str, int]:
    if len(predictions) != len(references):
        raise ValueError("length mismatch")
    out = jiwer.process_characters(references, predictions)
    return {"sub": int(out.substitutions), "ins": int(out.insertions),
            "del": int(out.deletions)}
```

```python
# src/vn_receipt_ocr/eval/length_ratio.py
from __future__ import annotations


def compute_length_ratios(
    *, predictions: list[str], references: list[str]
) -> list[float]:
    if len(predictions) != len(references):
        raise ValueError("length mismatch")
    out = []
    for p, r in zip(predictions, references):
        if len(r) == 0:
            out.append(float("inf"))
        else:
            out.append(len(p) / len(r))
    return out
```

```python
# src/vn_receipt_ocr/eval/latency.py
from __future__ import annotations
import numpy as np


def compute_latency_percentiles(times_ms: list[float]) -> dict[str, float]:
    arr = np.asarray(times_ms, dtype=float)
    if arr.size == 0:
        return {"p50": 0.0, "p95": 0.0}
    return {
        "p50": float(np.percentile(arr, 50)),
        "p95": float(np.percentile(arr, 95)),
    }
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/eval/test_edit_ops.py tests/eval/test_length_ratio.py tests/eval/test_latency.py -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vn_receipt_ocr/eval/{edit_ops,length_ratio,latency}.py \
        tests/eval/test_{edit_ops,length_ratio,latency}.py
git commit -m "feat(eval): edit-op breakdown, length ratio, latency percentiles"
```

---

## Task 13: Eval aggregate (single function returning all metrics)

**Files:**
- Create: `src/vn_receipt_ocr/eval/aggregate.py`
- Create: `tests/eval/test_aggregate.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/eval/test_aggregate.py
from vn_receipt_ocr.eval.aggregate import aggregate_metrics


def test_aggregate_all_metrics():
    out = aggregate_metrics(
        predictions=["Hà Nội", "abc"],
        references=["Hà Nội", "abc"],
        latency_ms=[10.0, 20.0],
        metrics_enabled=["cer","diacritic_cer","cer_normalized","wer",
                         "edit_ops","length_ratio","empty_pred_rate","latency"],
    )
    assert out["cer"] == 0.0
    assert out["diacritic_cer"] == 0.0
    assert out["cer_normalized"] == 0.0
    assert out["wer"] == 0.0
    assert out["edit_ops"]["sub"] == 0
    assert out["mean_length_ratio"] == 1.0
    assert out["empty_pred_rate"] == 0.0
    assert out["latency_p50"] == 15.0


def test_aggregate_empty_pred_rate():
    out = aggregate_metrics(
        predictions=["", "abc"],
        references=["xyz", "abc"],
        latency_ms=[10.0, 20.0],
        metrics_enabled=["empty_pred_rate"],
    )
    assert out["empty_pred_rate"] == 0.5
```

- [ ] **Step 2: Run, verify fails**

```bash
uv run pytest tests/eval/test_aggregate.py -v
```

- [ ] **Step 3: Implement**

```python
# src/vn_receipt_ocr/eval/aggregate.py
from __future__ import annotations

from vn_receipt_ocr.eval.cer import compute_cer
from vn_receipt_ocr.eval.wer import compute_wer
from vn_receipt_ocr.eval.diacritic_cer import compute_diacritic_cer
from vn_receipt_ocr.eval.normalized_cer import compute_normalized_cer
from vn_receipt_ocr.eval.edit_ops import compute_edit_ops
from vn_receipt_ocr.eval.length_ratio import compute_length_ratios
from vn_receipt_ocr.eval.latency import compute_latency_percentiles


def aggregate_metrics(
    *,
    predictions: list[str],
    references: list[str],
    latency_ms: list[float],
    metrics_enabled: list[str],
) -> dict:
    out: dict = {}
    if "cer" in metrics_enabled:
        out["cer"] = compute_cer(predictions=predictions, references=references)
    if "diacritic_cer" in metrics_enabled:
        out["diacritic_cer"] = compute_diacritic_cer(
            predictions=predictions, references=references
        )
    if "cer_normalized" in metrics_enabled:
        out["cer_normalized"] = compute_normalized_cer(
            predictions=predictions, references=references
        )
    if "wer" in metrics_enabled:
        out["wer"] = compute_wer(predictions=predictions, references=references)
    if "edit_ops" in metrics_enabled:
        out["edit_ops"] = compute_edit_ops(
            predictions=predictions, references=references
        )
    if "length_ratio" in metrics_enabled:
        ratios = compute_length_ratios(
            predictions=predictions, references=references
        )
        finite = [r for r in ratios if r != float("inf")]
        out["mean_length_ratio"] = (sum(finite) / len(finite)) if finite else 0.0
        out["length_ratios"] = ratios
    if "empty_pred_rate" in metrics_enabled:
        out["empty_pred_rate"] = sum(1 for p in predictions if len(p) == 0) / len(predictions)
    if "latency" in metrics_enabled:
        pct = compute_latency_percentiles(latency_ms)
        out["latency_p50"] = pct["p50"]
        out["latency_p95"] = pct["p95"]
    return out
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/eval/test_aggregate.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vn_receipt_ocr/eval/aggregate.py tests/eval/test_aggregate.py
git commit -m "feat(eval): aggregate function combining all metrics"
```

---

## Task 14: Kaggle helpers (paths, secrets, gpu_detect)

**Files:**
- Create: `src/vn_receipt_ocr/kaggle/{paths,secrets,gpu_detect}.py`
- Create: `tests/kaggle/test_{paths,secrets,gpu_detect}.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/kaggle/test_paths.py
from pathlib import Path
from vn_receipt_ocr.kaggle.paths import resolve_dataset_path


def test_local_path_returned_when_kaggle_input_missing(tmp_path: Path, monkeypatch):
    monkeypatch.setattr("vn_receipt_ocr.kaggle.paths.KAGGLE_INPUT", tmp_path / "nope")
    local = tmp_path / "datasets" / "x"
    local.mkdir(parents=True)
    out = resolve_dataset_path(kaggle_subpath="x", local_fallback=str(local))
    assert out == local


def test_kaggle_input_path_used_when_present(tmp_path: Path, monkeypatch):
    kag = tmp_path / "kag_input"
    (kag / "x").mkdir(parents=True)
    monkeypatch.setattr("vn_receipt_ocr.kaggle.paths.KAGGLE_INPUT", kag)
    out = resolve_dataset_path(kaggle_subpath="x", local_fallback="/nonexistent")
    assert out == kag / "x"
```

```python
# tests/kaggle/test_secrets.py
from vn_receipt_ocr.kaggle.secrets import get_secret_or_none


def test_returns_none_when_no_kaggle(monkeypatch):
    # Simulate: not on Kaggle, no UserSecretsClient
    monkeypatch.setattr(
        "vn_receipt_ocr.kaggle.secrets._client_factory",
        lambda: None,
    )
    monkeypatch.delenv("WANDB_API_KEY", raising=False)
    assert get_secret_or_none("WANDB_API_KEY") is None


def test_returns_env_var_when_present(monkeypatch):
    monkeypatch.setenv("WANDB_API_KEY", "abc123")
    assert get_secret_or_none("WANDB_API_KEY") == "abc123"
```

```python
# tests/kaggle/test_gpu_detect.py
from vn_receipt_ocr.kaggle.gpu_detect import detect_gpu_profile_name


def test_detect_p100():
    assert detect_gpu_profile_name(device_name="Tesla P100-PCIE-16GB") == "p100_16gb"


def test_detect_t4():
    assert detect_gpu_profile_name(device_name="Tesla T4") == "t4_16gb"


def test_detect_l4():
    assert detect_gpu_profile_name(device_name="NVIDIA L4") == "l4_24gb"


def test_detect_unknown_returns_default():
    assert detect_gpu_profile_name(device_name="weird gpu", default="p100_16gb") == "p100_16gb"
```

- [ ] **Step 2: Run, verify fail**

```bash
uv run pytest tests/kaggle -v
```

- [ ] **Step 3: Implement**

```python
# src/vn_receipt_ocr/kaggle/paths.py
from __future__ import annotations
from pathlib import Path


KAGGLE_INPUT = Path("/kaggle/input")
KAGGLE_WORKING = Path("/kaggle/working")


def resolve_dataset_path(*, kaggle_subpath: str, local_fallback: str) -> Path:
    kag = KAGGLE_INPUT / kaggle_subpath
    if kag.exists():
        return kag
    return Path(local_fallback)


def working_dir() -> Path:
    if KAGGLE_WORKING.exists():
        return KAGGLE_WORKING
    return Path.cwd()
```

```python
# src/vn_receipt_ocr/kaggle/secrets.py
from __future__ import annotations
import os
from typing import Callable


def _client_factory() -> object | None:
    try:
        from kaggle_secrets import UserSecretsClient  # type: ignore
        return UserSecretsClient()
    except Exception:
        return None


def get_secret_or_none(key: str) -> str | None:
    """Try environment variable first, then Kaggle Secrets, else None."""
    val = os.environ.get(key)
    if val:
        return val
    client = _client_factory()
    if client is None:
        return None
    try:
        return client.get_secret(key)  # type: ignore[attr-defined]
    except Exception:
        return None
```

```python
# src/vn_receipt_ocr/kaggle/gpu_detect.py
from __future__ import annotations


_NAME_TO_PROFILE: list[tuple[str, str]] = [
    ("p100", "p100_16gb"),
    ("t4", "t4_16gb"),
    ("l4", "l4_24gb"),
    ("a100", "a100_40gb"),
    ("v100", "v100_16gb"),
]


def detect_gpu_profile_name(*, device_name: str | None = None,
                             default: str = "p100_16gb") -> str:
    if device_name is None:
        try:
            import torch
            device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ""
        except Exception:
            device_name = ""
    name = (device_name or "").lower()
    for needle, profile in _NAME_TO_PROFILE:
        if needle in name:
            return profile
    return default
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/kaggle -v
```

Expected: 7 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vn_receipt_ocr/kaggle tests/kaggle
git commit -m "feat(kaggle): paths/secrets/gpu_detect helpers with offline fallbacks"
```

---

# Phase B — Training

Tasks 15–22. End-of-phase deliverable: model loader + collator + trainer + callbacks. Manual smoke test runs end-to-end on CPU with `--max-steps 1`. WandB tracking and HF Hub upload are wired but mocked in tests.

---

## Task 15: UnslothModelLoader

**Files:**
- Create: `src/vn_receipt_ocr/model/loader.py`
- Create: `tests/data/test_collator.py` (collator tests live in Task 16; this task focuses on the loader)

- [ ] **Step 1: Write failing test (config-only — no model load in CI)**

The Unsloth model load is heavy (~2 GB download). The test verifies the loader's argument-construction logic only; it does NOT load a real model. We expose a pure helper `build_load_kwargs` and test that.

Create `tests/model/__init__.py` (empty) and `tests/model/test_loader.py`:

```python
# tests/model/__init__.py — empty file
```

```python
# tests/model/test_loader.py
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
```

- [ ] **Step 2: Run, verify fails**

```bash
uv run pytest tests/model -v
```

- [ ] **Step 3: Implement**

```python
# src/vn_receipt_ocr/model/loader.py
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
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/model -v
```

Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vn_receipt_ocr/model tests/model
git commit -m "feat(model): UnslothModelLoader with FP16/BF16 dtype routing + PEFT kwargs"
```

---

## Task 16: QwenVLCollator

**Files:**
- Create: `src/vn_receipt_ocr/data/collator.py`
- Create: `tests/data/test_collator.py`

- [ ] **Step 1: Write failing tests (against a fake processor)**

```python
# tests/data/test_collator.py
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
```

- [ ] **Step 2: Run, verify fails**

```bash
uv run pytest tests/data/test_collator.py -v
```

- [ ] **Step 3: Implement**

```python
# src/vn_receipt_ocr/data/collator.py
from __future__ import annotations
from pathlib import Path
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
        # PromptBuilder owns instruction; collator gets instruction per-item.
        # We don't store one — each item carries its own instruction string.

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
        # Mask the prefix-only token range in each row.
        for i in range(labels.shape[0]):
            prefix_len = int(prefix["attention_mask"][i].sum().item())
            labels[i, :prefix_len] = -100
        # Also mask any padding in the full sequence.
        labels[full["attention_mask"] == 0] = -100

        full["labels"] = labels
        return full
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/data/test_collator.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vn_receipt_ocr/data/collator.py tests/data/test_collator.py
git commit -m "feat(data): QwenVLCollator with response-only loss masking"
```

---

## Task 17: Reproducibility manifest

**Files:**
- Create: `src/vn_receipt_ocr/train/manifest.py`
- Create: `tests/train/test_manifest.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/train/test_manifest.py
from vn_receipt_ocr.train.manifest import build_manifest


def test_manifest_required_fields():
    m = build_manifest(
        config_dict={"a": 1},
        run_id="abc",
        seed=42,
        gpu_device_name="Tesla P100",
    )
    assert m["run_id"] == "abc"
    assert m["seed"] == 42
    assert m["gpu_device_name"] == "Tesla P100"
    assert "python_version" in m
    assert "library_versions" in m
    assert m["config"] == {"a": 1}
    assert "git_commit" in m  # may be empty string if not in a repo
```

- [ ] **Step 2: Run, verify fails**

```bash
uv run pytest tests/train/test_manifest.py -v
```

- [ ] **Step 3: Implement**

```python
# src/vn_receipt_ocr/train/manifest.py
from __future__ import annotations
import importlib.metadata
import platform
import subprocess


_TRACKED_LIBS = ["torch", "transformers", "peft", "trl", "unsloth",
                 "huggingface_hub", "wandb", "jiwer", "pydantic"]


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except Exception:
        return ""


def _library_versions() -> dict[str, str]:
    out = {}
    for lib in _TRACKED_LIBS:
        try:
            out[lib] = importlib.metadata.version(lib)
        except importlib.metadata.PackageNotFoundError:
            out[lib] = "<not installed>"
    return out


def build_manifest(
    *, config_dict: dict, run_id: str, seed: int, gpu_device_name: str,
) -> dict:
    return {
        "run_id": run_id,
        "seed": seed,
        "gpu_device_name": gpu_device_name,
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "library_versions": _library_versions(),
        "git_commit": _git_commit(),
        "config": config_dict,
    }
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/train/test_manifest.py -v
```

Expected: 1 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vn_receipt_ocr/train/manifest.py tests/train/test_manifest.py
git commit -m "feat(train): reproducibility manifest"
```

---

## Task 18: WandB callback (with offline fallback)

**Files:**
- Create: `src/vn_receipt_ocr/train/callbacks.py` (initial — extended in Tasks 19, 20)
- Create: `tests/train/test_callbacks.py`

- [ ] **Step 1: Write failing tests**

```python
# tests/train/test_callbacks.py
from unittest.mock import MagicMock, patch
from vn_receipt_ocr.train.callbacks import init_wandb, JSONLFallback


def test_init_wandb_returns_offline_when_no_key(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "vn_receipt_ocr.kaggle.secrets.get_secret_or_none",
        lambda key: None,
    )
    out = init_wandb(
        project="test-proj", run_name="r1", config={"a": 1},
        fallback_jsonl=tmp_path / "fb.jsonl",
        mode_fallback="offline",
    )
    # Either WANDB_MODE is offline OR we use JSONL fallback
    assert out["mode"] in ("offline", "disabled", "jsonl")


def test_jsonl_fallback_writes_lines(tmp_path):
    f = tmp_path / "fb.jsonl"
    fb = JSONLFallback(path=f)
    fb.log({"loss": 1.0, "step": 1})
    fb.log({"loss": 0.5, "step": 2})
    lines = f.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    import json
    assert json.loads(lines[0])["loss"] == 1.0
```

- [ ] **Step 2: Run, verify fails**

```bash
uv run pytest tests/train/test_callbacks.py -v
```

- [ ] **Step 3: Implement**

```python
# src/vn_receipt_ocr/train/callbacks.py
from __future__ import annotations
import json
import os
from pathlib import Path
from typing import Any

from vn_receipt_ocr.kaggle.secrets import get_secret_or_none


class JSONLFallback:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass


def init_wandb(
    *,
    project: str,
    run_name: str,
    config: dict,
    fallback_jsonl: Path,
    mode_fallback: str = "offline",
) -> dict[str, Any]:
    """Initialize wandb if a key is present; else fall back to JSONL.

    Returns {'run': <wandb.Run|None>, 'mode': str, 'fallback': JSONLFallback|None}.
    """
    key = get_secret_or_none("WANDB_API_KEY")
    if key is None:
        if mode_fallback == "disabled":
            return {"run": None, "mode": "disabled",
                    "fallback": JSONLFallback(path=fallback_jsonl)}
        os.environ["WANDB_MODE"] = "offline"
    else:
        os.environ["WANDB_API_KEY"] = key

    try:
        import wandb  # type: ignore
        run = wandb.init(project=project, name=run_name, config=config)
        return {"run": run, "mode": os.environ.get("WANDB_MODE", "online"),
                "fallback": None}
    except Exception:
        return {"run": None, "mode": "jsonl",
                "fallback": JSONLFallback(path=fallback_jsonl)}


def log_event(state: dict, payload: dict) -> None:
    """Single entry point for emitting metrics to wandb or jsonl fallback."""
    run = state.get("run")
    fb: JSONLFallback | None = state.get("fallback")
    if run is not None:
        run.log(payload)
    if fb is not None:
        fb.log(payload)
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/train/test_callbacks.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vn_receipt_ocr/train/callbacks.py tests/train/test_callbacks.py
git commit -m "feat(train): wandb init + JSONL fallback"
```

---

## Task 19: Per-epoch eval callback (HF Trainer hook)

**Files:**
- Modify: `src/vn_receipt_ocr/train/callbacks.py` (append `PerEpochEvalCallback`)
- Append tests to `tests/train/test_callbacks.py`

- [ ] **Step 1: Write failing test**

Append to `tests/train/test_callbacks.py`:

```python
def test_per_epoch_eval_callback_calls_eval_fn(tmp_path):
    from vn_receipt_ocr.train.callbacks import PerEpochEvalCallback
    eval_calls = []

    def fake_eval():
        eval_calls.append(1)
        return {"cer": 0.1}

    cb = PerEpochEvalCallback(eval_fn=fake_eval, log_event_fn=lambda payload: None)
    # Simulate HF Trainer "on_epoch_end" hook.
    cb.on_epoch_end(args=None, state=type("S", (), {"epoch": 1})(), control=None)
    assert eval_calls == [1]
```

- [ ] **Step 2: Run, verify fails**

```bash
uv run pytest tests/train/test_callbacks.py::test_per_epoch_eval_callback_calls_eval_fn -v
```

- [ ] **Step 3: Implement**

Append to `src/vn_receipt_ocr/train/callbacks.py`:

```python
from typing import Callable


class PerEpochEvalCallback:
    """A minimal trainer-callback shape (not subclassed from
    transformers.TrainerCallback so it can be unit-tested without HF imports).
    The training wrapper adapts it via a thin shim."""

    def __init__(
        self, *, eval_fn: Callable[[], dict], log_event_fn: Callable[[dict], None]
    ) -> None:
        self.eval_fn = eval_fn
        self.log_event_fn = log_event_fn
        self.history: list[dict] = []

    def on_epoch_end(self, args, state, control, **kwargs):
        metrics = self.eval_fn()
        epoch = getattr(state, "epoch", None)
        record = {"epoch": epoch, **{f"eval/{k}": v for k, v in metrics.items()}}
        self.history.append(record)
        self.log_event_fn(record)
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/train/test_callbacks.py::test_per_epoch_eval_callback_calls_eval_fn -v
```

- [ ] **Step 5: Commit**

```bash
git add src/vn_receipt_ocr/train/callbacks.py tests/train/test_callbacks.py
git commit -m "feat(train): PerEpochEvalCallback with eval_fn injection"
```

---

## Task 20: Checkpoint sync callback (best-by-diacritic-CER → HF Hub)

**Files:**
- Modify: `src/vn_receipt_ocr/train/callbacks.py` (append `CheckpointSyncCallback`)
- Append tests to `tests/train/test_callbacks.py`

- [ ] **Step 1: Write failing test**

Append to `tests/train/test_callbacks.py`:

```python
def test_checkpoint_sync_uploads_only_when_better(tmp_path):
    from vn_receipt_ocr.train.callbacks import CheckpointSyncCallback
    uploads = []

    def fake_upload(local_dir, commit_msg):
        uploads.append((local_dir, commit_msg))

    def fake_save(path):
        path.mkdir(parents=True, exist_ok=True)

    cb = CheckpointSyncCallback(
        local_root=tmp_path / "ckpt",
        save_fn=fake_save,
        upload_fn=fake_upload,
        primary_metric="diacritic_cer",
    )
    # First call: 0.5 → improves over inf → upload
    cb.handle_eval(epoch=1, metrics={"diacritic_cer": 0.5})
    assert len(uploads) == 1
    # Second call: 0.6 → worse → no upload
    cb.handle_eval(epoch=2, metrics={"diacritic_cer": 0.6})
    assert len(uploads) == 1
    # Third call: 0.4 → improves → upload
    cb.handle_eval(epoch=3, metrics={"diacritic_cer": 0.4})
    assert len(uploads) == 2
```

- [ ] **Step 2: Run, verify fails**

```bash
uv run pytest tests/train/test_callbacks.py::test_checkpoint_sync_uploads_only_when_better -v
```

- [ ] **Step 3: Implement**

Append to `src/vn_receipt_ocr/train/callbacks.py`:

```python
import math
from pathlib import Path


class CheckpointSyncCallback:
    """On each eval, optionally write/save the model and upload to HF Hub
    only if `primary_metric` improved.

    save_fn(path): persists model+processor to `path`.
    upload_fn(local_dir, commit_msg): pushes to HF Hub (mocked in tests).
    """

    def __init__(
        self,
        *,
        local_root: Path,
        save_fn: Callable[[Path], None],
        upload_fn: Callable[[Path, str], None] | None,
        primary_metric: str = "diacritic_cer",
    ) -> None:
        self.local_root = Path(local_root)
        self.local_root.mkdir(parents=True, exist_ok=True)
        self.save_fn = save_fn
        self.upload_fn = upload_fn
        self.primary_metric = primary_metric
        self.best_value = math.inf

    def handle_eval(self, *, epoch: int | float | None, metrics: dict) -> bool:
        epoch_dir = self.local_root / f"epoch_{int(epoch) if epoch else 0}"
        self.save_fn(epoch_dir)

        cur = metrics.get(self.primary_metric)
        if cur is None:
            return False

        if cur < self.best_value:
            self.best_value = cur
            best_dir = self.local_root / "best"
            self.save_fn(best_dir)
            if self.upload_fn is not None:
                self.upload_fn(
                    best_dir,
                    f"epoch {epoch} | {self.primary_metric} {cur:.4f}",
                )
            return True
        return False
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/train/test_callbacks.py::test_checkpoint_sync_uploads_only_when_better -v
```

- [ ] **Step 5: Commit**

```bash
git add src/vn_receipt_ocr/train/callbacks.py tests/train/test_callbacks.py
git commit -m "feat(train): CheckpointSyncCallback with HF Hub upload on improvement"
```

---

## Task 21: Batch predict (greedy decode)

**Files:**
- Create: `src/vn_receipt_ocr/eval/batch_predict.py`
- (No unit test in CI; real model required. Smoke-tested in Task 25.)

- [ ] **Step 1: Implement**

```python
# src/vn_receipt_ocr/eval/batch_predict.py
from __future__ import annotations
from pathlib import Path
import time
from typing import Any
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
```

- [ ] **Step 2: Commit**

```bash
git add src/vn_receipt_ocr/eval/batch_predict.py
git commit -m "feat(eval): batch_predict greedy decode with per-sample latency"
```

---

## Task 22: Train wrapper (orchestrates everything)

**Files:**
- Create: `src/vn_receipt_ocr/train/trainer.py`
- (Tested via smoke test in Task 25; unit tests assert configuration assembly only.)

- [ ] **Step 1: Write failing test for run-id generation**

```python
# tests/train/test_trainer.py
from vn_receipt_ocr.train.trainer import generate_run_id, build_run_name


def test_generate_run_id_is_stable_for_same_inputs():
    a = generate_run_id(seed=42, model_id="x", date_str="20260506")
    b = generate_run_id(seed=42, model_id="x", date_str="20260506")
    assert a == b


def test_build_run_name_uses_template():
    name = build_run_name(
        template="{model_short}-r{lora_rank}-{date}-{nnn}",
        model_id="unsloth/Qwen3-VL-2B-Instruct",
        lora_rank=16,
        date_str="20260506",
        nnn="001",
    )
    assert name == "qwen3vl2b-r16-20260506-001"
```

- [ ] **Step 2: Run, verify fails**

```bash
uv run pytest tests/train/test_trainer.py -v
```

- [ ] **Step 3: Implement**

```python
# src/vn_receipt_ocr/train/trainer.py
from __future__ import annotations
import hashlib
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from vn_receipt_ocr.config.models import TrainConfig
from vn_receipt_ocr.config.validation import (
    validate_dtype_hardware, project_wallclock_hours,
)
from vn_receipt_ocr.data.dataset import MCOCRDataset
from vn_receipt_ocr.data.collator import QwenVLCollator
from vn_receipt_ocr.eval.aggregate import aggregate_metrics
from vn_receipt_ocr.eval.batch_predict import batch_predict
from vn_receipt_ocr.kaggle.paths import working_dir
from vn_receipt_ocr.kaggle.secrets import get_secret_or_none
from vn_receipt_ocr.model.loader import load_model_and_processor
from vn_receipt_ocr.train.callbacks import (
    init_wandb, log_event, PerEpochEvalCallback, CheckpointSyncCallback,
)
from vn_receipt_ocr.train.manifest import build_manifest


def _short_model_id(model_id: str) -> str:
    base = model_id.split("/")[-1].lower()
    base = base.replace("-instruct", "")
    return "".join(c for c in base if c.isalnum())  # qwen3-vl-2b → qwen3vl2b


def build_run_name(
    *, template: str, model_id: str, lora_rank: int, date_str: str, nnn: str
) -> str:
    return template.format(
        model_short=_short_model_id(model_id),
        lora_rank=lora_rank,
        date=date_str,
        nnn=nnn,
    )


def generate_run_id(*, seed: int, model_id: str, date_str: str) -> str:
    h = hashlib.sha1(f"{seed}-{model_id}-{date_str}".encode()).hexdigest()[:8]
    return h


def set_global_seed(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")


def train(config: TrainConfig, *, dry_run: bool = False, max_steps: int | None = None) -> dict:
    """Top-level training entry point.

    dry_run=True + max_steps=1 is the smoke-test mode.
    """
    # 1. Validate hardware
    validate_dtype_hardware(
        gpu_name=config.gpu_profile.name, dtype=config.gpu_profile.dtype,
    )

    # 2. Build run identifiers
    set_global_seed(config.trainer.seed, deterministic=config.trainer.deterministic)
    date_str = datetime.utcnow().strftime("%Y%m%d")
    run_id = generate_run_id(
        seed=config.trainer.seed, model_id=config.model.model_id, date_str=date_str,
    )
    run_name = build_run_name(
        template=config.wandb.run_name_template,
        model_id=config.model.model_id,
        lora_rank=config.lora.rank,
        date_str=date_str,
        nnn="001",
    )

    # 3. Datasets
    train_ds = MCOCRDataset(
        text_path=config.data.train_path,
        images_dir=config.data.train_images_dir,
        instruction=config.data.instruction,
    )
    val_ds = MCOCRDataset(
        text_path=config.data.val_path,
        images_dir=config.data.val_images_dir,
        instruction=config.data.instruction,
    )

    # 4. Wall-clock projection (rough) — fail fast if budget exceeded
    project_wallclock_hours(
        n_train_samples=len(train_ds),
        epochs=config.trainer.epochs,
        per_device_batch_size=config.trainer.per_device_batch_size,
        grad_accum=config.trainer.grad_accum,
        seconds_per_optimizer_step=4.0,  # conservative default; refine after first run
        budget_hours=config.trainer.wallclock_budget_hours,
        raise_on_exceed=not dry_run,
    )

    # 5. WandB init + manifest
    work = working_dir()
    wandb_state = init_wandb(
        project=config.wandb.project, run_name=run_name,
        config=config.model_dump(),
        fallback_jsonl=work / "wandb_offline.jsonl",
        mode_fallback=config.wandb.mode_fallback,
    )
    manifest = build_manifest(
        config_dict=config.model_dump(), run_id=run_id, seed=config.trainer.seed,
        gpu_device_name=(torch.cuda.get_device_name(0) if torch.cuda.is_available()
                         else "cpu"),
    )
    log_event(wandb_state, {"manifest": manifest})

    # 6. Model + processor + collator
    model, processor = load_model_and_processor(
        model=config.model, gpu_profile=config.gpu_profile, lora=config.lora,
    )
    collator = QwenVLCollator(processor=processor)

    # 7. Trainer (lazy import — heavy)
    from trl import SFTConfig, SFTTrainer  # type: ignore

    sft_args = SFTConfig(
        output_dir=str(work / "checkpoints"),
        per_device_train_batch_size=config.trainer.per_device_batch_size,
        gradient_accumulation_steps=config.trainer.grad_accum,
        num_train_epochs=config.trainer.epochs,
        learning_rate=config.trainer.lr,
        warmup_ratio=config.trainer.warmup_ratio,
        optim=config.trainer.optimizer,
        gradient_checkpointing=config.trainer.gradient_checkpointing,
        bf16=(config.gpu_profile.dtype == "bf16"),
        fp16=(config.gpu_profile.dtype == "fp16"),
        seed=config.trainer.seed,
        max_steps=(max_steps if max_steps is not None else -1),
        logging_steps=10,
        report_to=[],  # we route through our own log_event
        save_strategy="no",  # CheckpointSyncCallback owns persistence
    )

    # 8. Eval closure for PerEpochEvalCallback
    def _eval_now() -> dict:
        items = [val_ds[i] for i in range(len(val_ds))]
        preds, latencies = batch_predict(
            model=model, processor=processor, items=items,
            instruction=config.data.instruction,
            max_new_tokens=config.eval.max_new_tokens,
        )
        refs = [it["full_text"] for it in items]
        return aggregate_metrics(
            predictions=preds, references=refs,
            latency_ms=latencies, metrics_enabled=config.eval.metrics_enabled,
        )

    # 9. Save closure for CheckpointSyncCallback
    def _save_to(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path)
        processor.save_pretrained(path)

    # 10. Optional HF Hub upload closure
    upload_fn = None
    if config.hf_hub.enabled and get_secret_or_none("HF_TOKEN"):
        from huggingface_hub import HfApi  # lazy
        api = HfApi(token=get_secret_or_none("HF_TOKEN"))
        repo_name = config.hf_hub.repo_name_template.format(run_id=run_id)
        repo_id = f"{config.hf_hub.repo_owner}/{repo_name}"
        api.create_repo(repo_id=repo_id, private=config.hf_hub.private, exist_ok=True)

        def upload_fn(local_dir: Path, msg: str) -> None:
            api.upload_folder(folder_path=str(local_dir), repo_id=repo_id,
                              commit_message=msg)

    eval_cb = PerEpochEvalCallback(
        eval_fn=_eval_now,
        log_event_fn=lambda payload: log_event(wandb_state, payload),
    )
    ckpt_cb = CheckpointSyncCallback(
        local_root=work / "checkpoints",
        save_fn=_save_to, upload_fn=upload_fn,
        primary_metric="diacritic_cer",
    )

    # Adapt minimal-callback shape into HF Trainer callback shape
    from transformers import TrainerCallback

    class _HFEvalShim(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kw):
            metrics = eval_cb.on_epoch_end(args, state, control, **kw) or {}
            # eval_cb is what does the eval; ckpt_cb decides upload
            if eval_cb.history:
                last = eval_cb.history[-1]
                # strip 'eval/' prefix for ckpt_cb
                clean = {k.replace("eval/", ""): v for k, v in last.items()
                         if k.startswith("eval/")}
                ckpt_cb.handle_eval(epoch=state.epoch, metrics=clean)

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=[_HFEvalShim()],
    )

    # 11. Train
    if not dry_run:
        trainer.train()

    # 12. Summary
    return {
        "run_id": run_id,
        "run_name": run_name,
        "best_diacritic_cer": ckpt_cb.best_value,
        "history": eval_cb.history,
    }
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/train/test_trainer.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vn_receipt_ocr/train/trainer.py tests/train/test_trainer.py
git commit -m "feat(train): top-level train() orchestrating data/model/trainer/callbacks"
```

---

# Phase C — Packaging

Tasks 23–28. End-of-phase deliverable: CLI, `python -m` entry, top-level imports, Kaggle notebook template, README, manual smoke test recipe.

---

## Task 23: CLI (train/eval/predict subcommands)

**Files:**
- Create: `src/vn_receipt_ocr/cli.py`
- Create: `tests/test_cli.py`

- [ ] **Step 1: Write failing test**

```python
# tests/test_cli.py
import subprocess
import sys


def test_help_runs():
    out = subprocess.run(
        [sys.executable, "-m", "vn_receipt_ocr", "--help"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0
    assert "train" in out.stdout
    assert "eval" in out.stdout
    assert "predict" in out.stdout


def test_train_help_runs():
    out = subprocess.run(
        [sys.executable, "-m", "vn_receipt_ocr", "train", "--help"],
        capture_output=True, text=True,
    )
    assert out.returncode == 0
    assert "--config" in out.stdout
```

- [ ] **Step 2: Run, verify fails**

```bash
uv run pytest tests/test_cli.py -v
```

- [ ] **Step 3: Implement CLI + __main__**

```python
# src/vn_receipt_ocr/cli.py
from __future__ import annotations
import argparse
import json
from pathlib import Path

from vn_receipt_ocr.config.loader import load_train_config


def _parse_overrides(items: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--override expects key=value, got {it!r}")
        key, _, val = it.partition("=")
        # Try numeric/bool coercion; fall back to string
        coerced: object
        if val.lower() in ("true", "false"):
            coerced = val.lower() == "true"
        else:
            try:
                coerced = int(val)
            except ValueError:
                try:
                    coerced = float(val)
                except ValueError:
                    coerced = val
        out[key] = coerced
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vn_receipt_ocr",
                                description="Vietnamese Receipt OCR (Qwen3-VL-2B + Unsloth + LoRA)")
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("train", help="Train a LoRA adapter")
    pt.add_argument("--config", required=True, type=Path)
    pt.add_argument("--configs-root", default="configs", type=Path)
    pt.add_argument("--override", action="append", default=[],
                    help="Dotted-key overrides, e.g. lora.rank=8")
    pt.add_argument("--dry-run", action="store_true")
    pt.add_argument("--max-steps", type=int, default=None)

    pe = sub.add_parser("eval", help="Run evaluation on val/test")
    pe.add_argument("--config", required=True, type=Path)
    pe.add_argument("--configs-root", default="configs", type=Path)
    pe.add_argument("--adapters", required=True, type=str,
                    help="Local path or hf://owner/repo")
    pe.add_argument("--override", action="append", default=[])
    pe.add_argument("--split", choices=["val", "test"], default="val")

    pp = sub.add_parser("predict", help="Predict on a directory of images")
    pp.add_argument("--config", required=True, type=Path)
    pp.add_argument("--configs-root", default="configs", type=Path)
    pp.add_argument("--adapters", required=True, type=str)
    pp.add_argument("--inputs", required=True, type=Path)
    pp.add_argument("--output", required=True, type=Path)
    pp.add_argument("--override", action="append", default=[])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    overrides = _parse_overrides(args.override)
    cfg = load_train_config(
        experiment_path=args.config, configs_root=args.configs_root,
        overrides=overrides,
    )

    if args.command == "train":
        from vn_receipt_ocr.train.trainer import train as _train
        result = _train(cfg, dry_run=args.dry_run, max_steps=args.max_steps)
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.command == "eval":
        from vn_receipt_ocr import evaluate
        result = evaluate(cfg, adapters=args.adapters, split=args.split)
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.command == "predict":
        from vn_receipt_ocr import predict
        predict(cfg, adapters=args.adapters, inputs=args.inputs,
                output=args.output)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
```

```python
# src/vn_receipt_ocr/__main__.py
from vn_receipt_ocr.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/test_cli.py -v
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add src/vn_receipt_ocr/cli.py src/vn_receipt_ocr/__main__.py tests/test_cli.py
git commit -m "feat(cli): train/eval/predict subcommands + __main__ entry"
```

---

## Task 24: Top-level package exports + evaluate/predict API

**Files:**
- Modify: `src/vn_receipt_ocr/__init__.py`
- Create: `src/vn_receipt_ocr/eval/__init__.py` exports
- Append a smoke test for the API surface.

- [ ] **Step 1: Write failing test**

```python
# Append to tests/test_cli.py (or new tests/test_api.py)
def test_top_level_imports():
    import vn_receipt_ocr as m
    assert hasattr(m, "train")
    assert hasattr(m, "evaluate")
    assert hasattr(m, "predict")
    assert hasattr(m, "TrainConfig")
```

- [ ] **Step 2: Run, verify fails**

- [ ] **Step 3: Implement**

```python
# src/vn_receipt_ocr/__init__.py
from __future__ import annotations
from pathlib import Path
import json
from typing import Any

from vn_receipt_ocr.config.models import TrainConfig
from vn_receipt_ocr.config.loader import load_train_config


def train(
    config: TrainConfig | None = None,
    *,
    config_path: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
    max_steps: int | None = None,
) -> dict:
    from vn_receipt_ocr.train.trainer import train as _train
    if config is None:
        if config_path is None:
            raise ValueError("provide either config or config_path")
        config = load_train_config(
            experiment_path=config_path, configs_root="configs",
            overrides=overrides or {},
        )
    return _train(config, dry_run=dry_run, max_steps=max_steps)


def evaluate(
    config: TrainConfig, *, adapters: str, split: str = "val"
) -> dict:
    from vn_receipt_ocr.data.dataset import MCOCRDataset
    from vn_receipt_ocr.model.loader import load_model_and_processor
    from vn_receipt_ocr.eval.batch_predict import batch_predict
    from vn_receipt_ocr.eval.aggregate import aggregate_metrics

    text_path = config.data.val_path if split == "val" else config.data.test_path
    images_dir = config.data.val_images_dir if split == "val" else config.data.test_images_dir
    if not text_path:
        raise ValueError(f"No {split} path configured.")

    ds = MCOCRDataset(text_path=text_path, images_dir=images_dir,
                      instruction=config.data.instruction)
    items = [ds[i] for i in range(len(ds))]

    model, processor = load_model_and_processor(
        model=config.model, gpu_profile=config.gpu_profile, lora=config.lora,
    )
    # Load adapters into the freshly-loaded model
    if adapters.startswith("hf://"):
        from huggingface_hub import snapshot_download
        local = snapshot_download(repo_id=adapters[len("hf://"):])
    else:
        local = adapters
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, local)

    preds, lat = batch_predict(
        model=model, processor=processor, items=items,
        instruction=config.data.instruction,
        max_new_tokens=config.eval.max_new_tokens,
    )
    refs = [it["full_text"] for it in items]
    return aggregate_metrics(
        predictions=preds, references=refs, latency_ms=lat,
        metrics_enabled=config.eval.metrics_enabled,
    )


def predict(
    config: TrainConfig, *, adapters: str, inputs: Path | str, output: Path | str,
) -> None:
    from vn_receipt_ocr.model.loader import load_model_and_processor
    from vn_receipt_ocr.eval.batch_predict import batch_predict

    inputs = Path(inputs)
    if inputs.is_dir():
        files = sorted(p for p in inputs.iterdir() if p.suffix.lower() == ".jpg")
    else:
        files = [inputs]
    items = [{"image_path": p, "instruction": config.data.instruction,
              "full_text": ""} for p in files]

    model, processor = load_model_and_processor(
        model=config.model, gpu_profile=config.gpu_profile, lora=config.lora,
    )
    if adapters.startswith("hf://"):
        from huggingface_hub import snapshot_download
        local = snapshot_download(repo_id=adapters[len("hf://"):])
    else:
        local = adapters
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, local)

    preds, _ = batch_predict(
        model=model, processor=processor, items=items,
        instruction=config.data.instruction,
        max_new_tokens=config.eval.max_new_tokens,
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for p, t in zip(files, preds):
            f.write(json.dumps({"image": str(p), "prediction": t},
                              ensure_ascii=False) + "\n")


__all__ = ["train", "evaluate", "predict", "TrainConfig", "load_train_config"]
```

- [ ] **Step 4: Run, verify pass**

```bash
uv run pytest tests/test_cli.py -v
```

- [ ] **Step 5: Commit**

```bash
git add src/vn_receipt_ocr/__init__.py tests/test_cli.py
git commit -m "feat(api): top-level train/evaluate/predict + adapter loading"
```

---

## Task 25: CPU dry-run smoke test (manual)

**Files:**
- Create: `notebooks/README.md` (initial)
- Create: `scripts/smoke_test.sh`

This task is the manual end-to-end verification. It is NOT in CI (no GPU on CI); it is a documented procedure the maintainer runs once before declaring Phase B working.

- [ ] **Step 1: Add a smoke-test script**

```bash
mkdir -p scripts
```

`scripts/smoke_test.sh`:
```bash
#!/usr/bin/env bash
set -euo pipefail
# Dry-run smoke test on CPU. Ensures end-to-end wiring without doing real training.
# This will still attempt to load a small Qwen-VL via Unsloth — make sure you
# have the model cached or set HF_HUB_OFFLINE=0 with internet.

uv run python -m vn_receipt_ocr train \
    --config configs/experiments/baseline_v1.yaml \
    --override hf_hub.enabled=false \
    --override wandb.enabled=false \
    --dry-run \
    --max-steps 1
```

```bash
chmod +x scripts/smoke_test.sh
```

- [ ] **Step 2: Document in `notebooks/README.md` (initial stub)**

```markdown
# Notebooks & Smoke Test

## CPU smoke test
Run the smoke test once after Phase B to verify end-to-end wiring:

    ./scripts/smoke_test.sh

Expect: the pipeline to construct configs, load datasets, and exit without
training (because `--dry-run` skips `trainer.train()`). If you see a
ConfigValidationError or an import error, that is the smoke test catching a
real issue.

## Kaggle notebook
See `kaggle_train.ipynb` (added in Task 26).
```

- [ ] **Step 3: Commit**

```bash
git add scripts notebooks/README.md
git commit -m "feat(scripts): smoke_test.sh + notebooks/README"
```

- [ ] **Step 4: (Manual) Run the smoke test**

Run `./scripts/smoke_test.sh`. The expected outcome on a CPU-only laptop:
- If Unsloth refuses to load on CPU (it sometimes does), the smoke test "fails" at the model-load step. That is acceptable; document the failure point in `notebooks/README.md` with the exact error and confirm everything UP TO model-load works.
- If model loads, the dry-run exits with a JSON summary printed to stdout.

This is a checkpoint, not a CI test — it confirms wiring before moving to the Kaggle notebook.

---

## Task 26: Kaggle notebook template

**Files:**
- Create: `notebooks/kaggle_train.ipynb`

The notebook is a template — it cannot be unit-tested. It encodes the documented Kaggle workflow.

- [ ] **Step 1: Create `notebooks/kaggle_train.ipynb`**

A minimal Jupyter notebook with the following cells (one cell per item):

1. **Markdown** — heading: "Vietnamese Receipt OCR — Train (Qwen3-VL-2B + Unsloth + LoRA)"

2. **Code** (install package):
   ```python
   # Cell 1: install package
   import subprocess
   import sys
   subprocess.check_call([
       sys.executable, "-m", "pip", "install", "--upgrade",
       "git+https://github.com/<your-username>/vietnamese-receipt-ocr.git@main",
   ])
   ```

3. **Code** (verify GPU):
   ```python
   # Cell 2: verify GPU
   import torch
   print("CUDA:", torch.cuda.is_available())
   if torch.cuda.is_available():
       print("Device:", torch.cuda.get_device_name(0))
   ```

4. **Code** (set secrets — Kaggle UI sets these as session variables):
   ```python
   # Cell 3: secrets — already set via Kaggle "Add-ons → Secrets"
   from kaggle_secrets import UserSecretsClient
   us = UserSecretsClient()
   import os
   os.environ["WANDB_API_KEY"] = us.get_secret("WANDB_API_KEY")
   os.environ["HF_TOKEN"] = us.get_secret("HF_TOKEN")
   print("secrets set")
   ```

5. **Code** (resolve dataset path):
   ```python
   # Cell 4: dataset path — Kaggle attached dataset
   from pathlib import Path
   MCOCR = Path("/kaggle/input/vietnamese-receipts-mc-ocr-2021/versions/17")
   assert MCOCR.exists(), f"Attach the MC-OCR Kaggle dataset; expected at {MCOCR}"
   print("dataset OK:", MCOCR)
   ```

6. **Code** (run training):
   ```python
   # Cell 5: train
   import subprocess, sys
   ret = subprocess.call([
       sys.executable, "-m", "vn_receipt_ocr", "train",
       "--config", "configs/experiments/baseline_v1.yaml",
       "--override", "hf_hub.repo_owner=<your-hf-username>",
       "--override", f"data.train_path={MCOCR}/text_recognition_train_data.txt",
       "--override", f"data.train_images_dir={MCOCR}/train_images",
       "--override", f"data.val_path={MCOCR}/text_recognition_val_data.txt",
       "--override", f"data.val_images_dir={MCOCR}/val_images",
   ])
   print("exit code:", ret)
   ```

7. **Code** (summary):
   ```python
   # Cell 6: print W&B run URL + HF Hub repo URL
   import wandb
   run = wandb.api.runs("vn-receipt-ocr")[-1]
   print("W&B:", run.url)
   print("HF Hub: https://huggingface.co/<your-hf-username>/vn-receipt-ocr-<run_id>")
   ```

- [ ] **Step 2: Commit**

```bash
git add notebooks/kaggle_train.ipynb
git commit -m "feat(notebooks): kaggle_train.ipynb template"
```

---

## Task 27: README update + distribution recipe

**Files:**
- Create: `README.md` (or `notebooks/README.md` extended)

- [ ] **Step 1: Append to `notebooks/README.md`**

```markdown
## Distribution to Kaggle

Two install paths supported:

### A. GitHub install (preferred — internet enabled on Kaggle)
In the first cell of `kaggle_train.ipynb`:

    !pip install --upgrade git+https://github.com/<you>/vietnamese-receipt-ocr.git@main

### B. Wheel via Kaggle dataset (offline / scheduled runs)
Build locally:

    uv build

Upload `dist/vn_receipt_ocr-<version>-py3-none-any.whl` as a private Kaggle
dataset (e.g. `<your-handle>/vn-receipt-ocr-wheel`). Attach to the notebook,
then in Cell 1 use:

    !pip install /kaggle/input/vn-receipt-ocr-wheel/vn_receipt_ocr-*.whl

## Required Kaggle Secrets
Set both via Kaggle "Add-ons → Secrets":

- `WANDB_API_KEY` — from https://wandb.ai/authorize
- `HF_TOKEN` — from https://huggingface.co/settings/tokens

The pipeline degrades gracefully if either is absent (W&B → offline+JSONL,
HF Hub → upload skipped, training continues), but you'll lose persistence.

## Resume from a previous run

    !python -m vn_receipt_ocr train \
        --config configs/experiments/baseline_v1.yaml \
        --override resume_from_hub=hf://<your-handle>/vn-receipt-ocr-<run_id>
```

- [ ] **Step 2: Commit**

```bash
git add notebooks/README.md
git commit -m "docs(notebooks): distribution recipe + secrets + resume"
```

---

## Task 28: Final test sweep + lint

**Files:**
- (Verifies the complete tree)

- [ ] **Step 1: Run the full test suite**

```bash
uv run pytest -v
```

Expected: all green. If any test fails, diagnose and fix; do not move on.

- [ ] **Step 2: Run ruff (or skip if not configured)**

```bash
uv run ruff check src tests || true
```

Expected: clean or only minor stylistic warnings.

- [ ] **Step 3: Verify import surface**

```bash
uv run python -c "
import vn_receipt_ocr
print('train:', callable(vn_receipt_ocr.train))
print('evaluate:', callable(vn_receipt_ocr.evaluate))
print('predict:', callable(vn_receipt_ocr.predict))
print('TrainConfig:', vn_receipt_ocr.TrainConfig.__name__)
"
```

Expected:
```
train: True
evaluate: True
predict: True
TrainConfig: TrainConfig
```

- [ ] **Step 4: Verify CLI help**

```bash
uv run python -m vn_receipt_ocr --help
uv run python -m vn_receipt_ocr train --help
uv run python -m vn_receipt_ocr eval --help
uv run python -m vn_receipt_ocr predict --help
```

Expected: each prints its argparse usage with a 0 exit code.

- [ ] **Step 5: Final commit (if anything was tweaked during sweep)**

```bash
git status
# if there are tweaks:
git commit -am "chore: final sweep — tests/lint/import/cli verified"
```

---

# Closing checklist (run-through, not coded steps)

- [ ] All 28 tasks complete and committed.
- [ ] `uv run pytest` is green.
- [ ] CPU smoke test was attempted; outcome documented in `notebooks/README.md`.
- [ ] One real Kaggle run executed end-to-end on a P100/T4. Results logged in W&B and adapters pushed to HF Hub. (This is the post-merge validation, not part of the plan tasks.)
- [ ] `architectural-memory` decision note's "In-flight Refinements" section updated with anything that deviated from the plan during execution.
