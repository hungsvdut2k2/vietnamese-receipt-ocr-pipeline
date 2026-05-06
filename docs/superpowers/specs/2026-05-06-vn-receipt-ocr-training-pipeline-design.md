# Vietnamese Receipt OCR Training Pipeline — Design Spec

**Date:** 2026-05-06
**Topic:** LoRA fine-tuning pipeline for Qwen3-VL-2B on Vietnamese receipts, packaged for Kaggle GPUs.
**Predecessor:** [2026-05-06-eda-pipeline-design.md](./2026-05-06-eda-pipeline-design.md) (Phase 0 EDA — outputs consumed here as configuration inputs).

---

## §1. Goal and Non-Goals

### Goal
A Python package `vn_receipt_ocr`, installable on Kaggle, that LoRA-fine-tunes Qwen3-VL-2B via Unsloth to perform whole-document Vietnamese receipt transcription. A fixed-per-run Vietnamese instruction (configurable for paraphrase ablations) is paired with a receipt image; the model emits multi-line plain text. Trained on MC-OCR line-OCR concatenations; evaluated with text-similarity metrics; tracked in WandB; trained adapters pushed to Hugging Face Hub.

### Non-Goals (v1)
- No KIE / structured JSON output. The output is plain text.
- No multi-question VQA. One fixed instruction per sample.
- No augmentation. Skipped for the v1 baseline; add only as an ablation if v1 underperforms on rotated/blurred subsets.
- No multi-GPU training. Single-GPU baseline (P100/T4 16 GB); GPU profile YAML allows scaling up later without code change.
- No Viet-Receipt-VQA integration. Whole-document transcription does not match a VQA dataset's natural shape; deferred.
- No HTTP/gRPC serving. A batch `predict` CLI/API for local inference is included (it shares code with `eval`), but no server, no Gradio app, no Triton wrapper.
- No real-world held-out test set integration. Pipeline takes a path; user provides test data later.

### Success criteria for the pipeline (not the model)
- A Kaggle notebook runs `pip install` + one CLI invocation and produces a trained adapter on Hugging Face Hub plus a complete W&B run.
- Resuming from a partial run is one config flag.
- Swapping GPU profile (P100 → L4) requires only a config change.
- All eval metrics are reproducible from a saved adapter and config — no hidden state.

---

## §2. Subpackage Architecture

The package lives under `src/vn_receipt_ocr/` in the existing repository (extends the current `vietnamese-receipt-ocr` project, does not start a new project).

```
src/vn_receipt_ocr/
├── config/      Pydantic v2 config models + YAML loaders. One root config (TrainConfig) composing
│                ModelConfig, DataConfig, TrainerConfig, LoRAConfig, GPUProfileConfig,
│                WandBConfig, HFHubConfig. Validation at load time, fail-fast.
├── data/        MCOCRDataset (reads text_recognition_*.txt, groups by prefix, sorts by index,
│                joins with \n, pairs with image from train_images/ or val_images/).
│                PromptBuilder (fixed Vietnamese instruction string).
│                QwenVLCollator (Unsloth-compatible; chat template applied; image preprocessing
│                per Qwen3-VL spec; pad/mask labels for instruction tokens so loss is response-only).
├── model/       UnslothModelLoader (FastVisionModel.from_pretrained, applies LoRA via
│                FastVisionModel.get_peft_model; resolves model id from config; handles
│                4bit/bf16/fp16 dtype via GPU profile).
├── train/       Trainer wrapper (Unsloth's SFTTrainer wrapping HF Trainer per Unsloth recommendation
│                for VLMs). Callbacks: WandBCallback, PerEpochEvalCallback, CheckpointSyncCallback.
├── eval/        cer.py (jiwer-based CER), wer.py, diacritic_cer.py (NFC-decomposed Vietnamese
│                diacritic char filter), normalized_cer.py (applies normalization_rules.yaml before
│                CER), edit_ops.py (sub/ins/del breakdown), length_ratio.py, latency.py,
│                batch_predict.py (greedy decode, batched).
├── kaggle/      paths.py (resolves /kaggle/input/.../ vs local paths transparently).
│                secrets.py (UserSecretsClient wrapper with offline fallback).
│                gpu_detect.py (auto-selects GPU profile from torch.cuda.get_device_name).
├── cli.py       argparse → TrainConfig → train()/evaluate()/predict(). Subcommands: train, eval, predict.
└── __main__.py  Enables `python -m vn_receipt_ocr <subcommand>`.

configs/
├── gpu_profiles/ p100_16gb.yaml, t4_16gb.yaml, t4x2_32gb.yaml, l4_24gb.yaml
├── data/         mcocr_train_val.yaml (paths, prompt template)
├── model/        qwen3_vl_2b.yaml, qwen2_vl_2b.yaml (fallback / iteration aid)
├── lora/         r16_attn_mlp.yaml (default), r8_attn_only.yaml, r32_attn_mlp.yaml
└── experiments/  baseline_v1.yaml (composes one of each above)

notebooks/
└── kaggle_train.ipynb  (~10 cells; install package, set secrets, run CLI, tail logs)

tests/             Unit tests for: data loader (line-grouping, prefix-strip, index-sort, join),
                   metrics (CER, diacritic-CER, normalized-CER on hand-crafted fixtures),
                   config validation (paths, dtype/hardware compatibility), Kaggle secret fallback.
                   Smoke test: 2 dummy samples, max_steps=1, on CPU.
```

### Module responsibilities (one-liner each)
- `config/`: validate every input before any work happens.
- `data/`: deterministic mapping from raw text_recognition files to (image, instruction, target) tuples.
- `model/`: resolve and load the right Qwen-VL variant + LoRA wrapper for a given GPU profile.
- `train/`: orchestrate the SFT loop and emit signals via callbacks (no business logic in callbacks themselves).
- `eval/`: produce all metrics from (predictions, ground_truths) pairs — pure functions, no I/O.
- `kaggle/`: isolate every Kaggle-specific environment quirk so the rest of the package is environment-agnostic.

---

## §3. Data Sources and Splits

### Sources (v1)
- **Train**: `datasets/kagglehub/datasets/domixi1989/vietnamese-receipts-mc-ocr-2021/versions/17/text_recognition_train_data.txt` — 922 receipts after grouping by prefix.
- **Val**: `datasets/kagglehub/datasets/domixi1989/vietnamese-receipts-mc-ocr-2021/versions/17/text_recognition_val_data.txt` — 231 receipts after grouping by prefix.
- **Test**: TBD — user will provide later. Pipeline accepts the same line-OCR text-file format and grouping convention.
- **Images**: `train_images/<prefix>.jpg` and `val_images/<prefix>.jpg` from the same dataset version.

### Target reconstruction (deterministic procedure for every split)
Each row in the line-OCR file is `<filename>_<N>.jpg\t<line_text>`. To get a per-receipt full-text target:

1. Strip `_<N>.jpg` suffix → group key (the prefix, e.g. `mcocr_public_145014ftvdj`).
2. Within group, sort by integer `<N>`.
3. Join `<line_text>` values with `\n`.
4. Pair with the receipt image at `<split>_images/<prefix>.jpg`.

This must be the **same procedure for train, val, and test**. (Architectural invariant: train/val/test target shape consistency. Mixing this procedure with `mcocr_train_df.csv`'s `anno_texts` would create distribution mismatch — `anno_texts` only contains field-annotated regions, not all transcribed lines.)

### What we do NOT use
- `mcocr_train_df.csv` / `mcocr_val_sample_df.csv` (KIE-style field annotations) — irrelevant for whole-document transcription and would break target-shape consistency.
- 5CD-AI/Viet-Receipt-VQA — VQA dataset, no whole-document text field. Deferred.
- `text_recognition_*` line-crop sub-images as separate samples — different task (single-line OCR). Deferred.

### Train/val statistics (from preliminary scan during brainstorm)
- 922 train receipts, median 5 lines, P95 8 lines, max 29 lines.
- 231 val receipts.
- All within `max_seq_length = 203` tokens from `eda_outputs/training_caps.yaml` (P99 = 185).

---

## §4. Data Flow (Single Sample)

```
text_recognition_train_data.txt row → MCOCRDataset.__getitem__:
  1. (one-time on init) parse all rows, group by stripped-suffix prefix, sort within group
     by integer suffix, build {prefix → [line_text, ...]} index. Cache as attribute.
  2. for index i, look up prefix → join line_text list with \n → "full_text" target.
  3. open <split>_images/<prefix>.jpg → PIL → resize: longest_side=888 (target_resolution
     from training_caps.yaml), preserving aspect ratio, hard cap at longest_side=1388 (P95).
  4. PromptBuilder.build(image, instruction, full_text) → Qwen chat-template messages:
        [
          {"role": "user",
           "content": [
             {"type": "image", "image": pil},
             {"type": "text",  "text":  INSTRUCTION},
           ]},
          {"role": "assistant", "content": full_text},
        ]
  5. Apply Qwen3-VL processor → input_ids, attention_mask, pixel_values, image_grid_thw.
  6. Build labels: clone input_ids, mask all positions before the assistant turn to -100
     (loss only on the response). Image tokens already correctly handled by the processor.
  → return dict to QwenVLCollator, which pads to max-in-batch and stacks tensors.
```

### Fixed instruction
```
Trích xuất toàn bộ nội dung văn bản từ hóa đơn này, giữ nguyên thứ tự đọc từ trên xuống dưới.
```

(English gloss: "Extract all text content from this receipt, preserving reading order from top to bottom.")

The instruction is a config field (`DataConfig.instruction`), so paraphrase ablations are a config swap.

---

## §5. Training Loop

### Model loading (model/)
- `FastVisionModel.from_pretrained(model_id, load_in_4bit=False, dtype=<gpu_profile.dtype>)`.
- BF16 on T4/L4, FP16 on P100 (P100 lacks BF16). The GPU profile YAML encodes this; config validation rejects BF16 on P100 with a clear error pointing at the profile.
- Vision tower: frozen by default. Maps to Unsloth's `FastVisionModel.get_peft_model(..., finetune_vision_layers=False, finetune_language_layers=True, finetune_attention_modules=True, finetune_mlp_modules=True)`. Exposed as `lora.finetune_vision_layers` config flag for ablations.

### LoRA configuration (default)
- Rank 16, alpha 32, dropout 0.05.
- Target modules: `["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]` (attention + MLP per plan.md §4).
- Bias: `none`.
- Saved under `configs/lora/r16_attn_mlp.yaml`. Ablations as `r8_attn_only.yaml` and `r32_attn_mlp.yaml`.

### Trainer (train/)
- Unsloth `SFTTrainer` (HF Trainer wrapper).
- Per-device batch size 1 + gradient accumulation 8 (effective batch 8) on P100/T4.
- Gradient checkpointing on (`use_gradient_checkpointing="unsloth"`).
- AdamW 8-bit optimizer (Unsloth default), LR 1e-4, cosine schedule, warmup ratio 0.05.
- Default 3 epochs. `max_steps` automatically computed from `steps_per_epoch × epochs`; if the projected wall-clock exceeds the configured Kaggle wall-clock budget (default 8h to leave headroom under Kaggle's 9h Save-and-Run-All / 12h interactive limits), training fails fast at config-validation time with a message suggesting reduced epochs or `max_steps`.

### Callbacks
- **WandBCallback**: step-level loss / grad_norm / lr; every 50 steps GPU memory; epoch-level eval metrics; per-epoch sample-prediction Table.
- **PerEpochEvalCallback**: at each epoch end, runs `eval/batch_predict.py` greedy decode on the val split, computes the full metric suite (§6), passes results to WandB and checkpoint sync.
- **CheckpointSyncCallback**: writes adapters to `/kaggle/working/checkpoints/epoch_N/`; if eval `diacritic_cer` improved over previous best, also writes to `/kaggle/working/checkpoints/best/` and uploads to Hugging Face Hub via `huggingface_hub.upload_folder`.

### Reproducibility
- Top-level `seed: 42` propagated to torch / numpy / random / transformers / Unsloth.
- Deterministic-mode flag in config (default off — Unsloth's CUDA kernels are not fully deterministic, and forcing determinism slows training; flag exists for ablation comparisons).
- Per-run reproducibility manifest: full config YAML + Python version + key library versions + git commit hash + seed → uploaded to W&B and committed to the HF Hub repo.

---

## §6. Evaluation Contract

### Primary metrics (decision-driving)
- `cer` — overall character error rate (jiwer).
- `diacritic_cer` — CER restricted to characters whose NFD decomposition contains a Vietnamese diacritic combining mark (acute U+0301, grave U+0300, hook-above U+0309, tilde U+0303, dot-below U+0323, plus circumflex U+0302, breve U+0306, horn U+031B as base modifiers). The headline metric for plan.md §9 trigger criteria (diacritic CER > 5% triggers Phase 2 expansion). Implementation: `unicodedata.normalize("NFD", s)` → filter to characters whose decomposition intersects this set → re-compose → CER.
- `cer_normalized` — CER computed after applying `eda_outputs/normalization_rules.yaml` (numeric / date / currency canonicalization) to both prediction and ground truth. Distinguishes "format" errors from "semantic" errors.

### Secondary metrics (diagnostic)
- `wer` — word error rate.
- `edit_ops_sub`, `edit_ops_ins`, `edit_ops_del` — Levenshtein operation breakdown.
- `length_ratio_histogram` — distribution of `len(pred) / len(gt)` over the eval split. Logged as W&B histogram per epoch.
- `empty_prediction_rate` — fraction of samples with empty output.
- `latency_p50`, `latency_p95` — per-sample inference latency for the comparison table (plan.md §6 / slide 12).

### Qualitative
- Per-epoch W&B Table of 10 sample predictions (deterministic indices for reproducibility) with image, ground truth, prediction, per-sample CER.

### Decoding
- Greedy decode for v1.
- `max_new_tokens` = `training_caps.yaml.max_seq_length × 1.2` = 244.
- Beam search is a config-toggleable extension; deferred unless v1 hallucinates.

### Eval-only mode
`python -m vn_receipt_ocr eval --config <path> --adapters <path-or-hf-repo>` runs the same metric suite on the val (or any split path) without training. Reproduces W&B metrics and writes a JSON report.

### Excluded metrics (with reasoning)
- **Exact match**: will be ~0% on multi-line transcription; uninformative noise.
- **BLEU / ROUGE**: translation/summarization metrics; misleading for OCR-style output.
- **Per-field F1 / JSON validity / hallucination rate from plan.md §5**: do not apply — output is unstructured text, not JSON.
- **Bbox / grounding metrics from plan.md §5 Layer 4**: do not apply — output is text only, no bounding boxes.

---

## §7. WandB Integration

### Identification
- Project name: `vn-receipt-ocr` (single project; runs are differentiated by name).
- Run name template: `{model_short}-r{lora_rank}-{YYYYMMDD}-{nnn}` (e.g. `qwen3vl2b-r16-20260506-001`). `{nnn}` is a monotonic per-day counter.
- Run config: full TrainConfig as a flat dict, plus git commit hash and library versions.

### Auth and fallback
- `WANDB_API_KEY` read from Kaggle Secrets (`UserSecretsClient`).
- Missing key → `WANDB_MODE=offline` and JSONL fallback to `/kaggle/working/wandb_offline.jsonl`. Never raises.
- WandB outage at runtime → caught, logged locally, never propagated.

### Logged signals
- Step-level: `train/loss`, `train/grad_norm`, `train/lr`.
- Every 50 steps: `gpu/mem_allocated_gb`.
- Epoch-level: every metric from §6.
- Tables: per-epoch sample-prediction table (10 rows: image, ground_truth, prediction, per-sample CER).
- Artifacts: best LoRA adapters (~50–200 MB) at end of run; final config YAML; per-run reproducibility manifest.

---

## §8. Checkpointing and Hugging Face Hub

### Live checkpoints
- `/kaggle/working/checkpoints/epoch_N/` per-epoch.
- `/kaggle/working/checkpoints/best/` for the best (by `eval/diacritic_cer`).
- Format: Unsloth adapter safetensors + processor config.
- Note: `/kaggle/working/` is wiped after the kernel session ends; live checkpoints exist only for resume *within* a session.

### Hugging Face Hub upload
- One private model repository per run, named `<hf_user>/vn-receipt-ocr-{run_id}`.
- Triggered on each new best by `eval/diacritic_cer`.
- `huggingface_hub.upload_folder` with `commit_message=f"epoch {N} | diacritic_cer {x:.4f}"`.
- `HF_TOKEN` from Kaggle Secrets; missing token aborts the upload (with a clear error) but does not crash training — local checkpoint still exists for the session.
- Final checkpoint also pushed at end of run, plus a `README.md` containing config snapshot and final eval table.

### Resume
- Default: train from scratch.
- `train(..., resume_from_local=True)` checks `/kaggle/working/checkpoints/best/` first.
- `train(..., resume_from_hub="<repo>")` pulls adapters from a specified HF repo.
- Resume restores model weights and optimizer state if available; epoch counter resets unless explicitly continued.

### W&B artifact mirror
- Best adapter also uploaded as a W&B artifact at end of run. Redundant but cheap.

---

## §9. Configuration System

### Models (Pydantic v2)
- `TrainConfig` — root.
  - `model: ModelConfig` — `model_id`, `dtype`, `freeze_vision_tower`.
  - `lora: LoRAConfig` — `rank`, `alpha`, `dropout`, `target_modules`, `bias`.
  - `data: DataConfig` — train/val paths, `instruction`, `target_resolution`, `max_resolution`, `max_seq_length`.
  - `trainer: TrainerConfig` — `epochs`, `per_device_batch_size`, `grad_accum`, `lr`, `warmup_ratio`, `optimizer`, `gradient_checkpointing`, `seed`, `deterministic`.
  - `gpu_profile: GPUProfileConfig` — `name`, `dtype`, `vram_gb`, `recommended_batch_size`.
  - `wandb: WandBConfig` — `project`, `run_name_template`, `enabled`, `mode_fallback`.
  - `hf_hub: HFHubConfig` — `repo_owner`, `repo_name_template`, `private`, `enabled`.
  - `eval: EvalConfig` — `metrics_enabled`, `decode`, `max_new_tokens`, `sample_table_size`.

### Composition
- Configs compose via deep-merge. Precedence (low → high): `gpu_profiles/<profile>.yaml` < `data/<dataset>.yaml` < `model/<model>.yaml` < `lora/<lora>.yaml` < `experiments/<exp>.yaml` < CLI `--override key=value`.
- An experiment YAML is the authoritative entry point and references its component files by relative path.

### Validation (fail-fast)
- All paths exist or are markedly TBD (test path).
- Hardware/dtype compatibility (BF16 rejected on P100).
- Numeric ranges (LR > 0, rank > 0, etc.).
- Wall-clock projection: if `epochs × steps_per_epoch × estimated_step_time > 9 hours`, refuse with a suggestion.

### GPU profile auto-detection
- `vn_receipt_ocr.kaggle.gpu_detect.detect()` reads `torch.cuda.get_device_name(0)` and returns the matching profile name.
- If `--gpu-profile` CLI arg is missing, auto-detect is invoked and the chosen profile is logged.
- Fallback: P100 profile if no match.

---

## §10. Kaggle Entry Points

### Notebook template (`notebooks/kaggle_train.ipynb`, ~10 cells)
1. Cell 1: enable internet on the kernel.
2. Cell 2: `!pip install git+https://github.com/<you>/vietnamese-receipt-ocr.git@<branch>`. (Fallback documented in `notebooks/README.md`: `pip install /kaggle/input/<wheel-dataset>/*.whl`.)
3. Cell 3: attach MC-OCR Kaggle dataset; verify `/kaggle/input/vietnamese-receipts-mc-ocr-2021/`.
4. Cell 4: import package, verify GPU detection, print resolved config.
5. Cell 5: `!python -m vn_receipt_ocr train --config configs/experiments/baseline_v1.yaml`.
6. Cell 6: tail-end summary — best CER / diacritic-CER, HF Hub repo URL, W&B run URL.

### CLI
```
python -m vn_receipt_ocr train   --config <path> [--override key=value ...]
python -m vn_receipt_ocr eval    --config <path> --adapters <path-or-hf-repo>
python -m vn_receipt_ocr predict --config <path> --adapters <ref> --inputs <dir-or-file>
```

### Python API
```python
from vn_receipt_ocr import train, evaluate, predict
result = train(config_path="configs/experiments/baseline_v1.yaml",
               overrides={"trainer.epochs": 5})
metrics = evaluate(config_path="...", adapters="hf://<user>/vn-receipt-ocr-<run>")
preds = predict(config_path="...", adapters="...", inputs="path/to/images/")
```

CLI is a thin argparse wrapper over the Python API.

---

## §11. Reliability, Testing, Error Handling

### Fail-fast policy
- Configuration errors raised at load time, before any data or model is touched.
- Missing secret + non-offline mode → raise.
- HF upload failure → log, do not crash training.
- WandB outage → fall back to JSONL, do not crash.

### Tests (pytest)
- **Data loader**: prefix-strip on edge cases (multi-digit `_N`, hyphens in prefix), index-sort correctness, line-join with `\n` on hand-crafted fixtures, missing image handling.
- **Metrics**: CER / diacritic-CER / normalized-CER on hand-crafted (pred, gt) pairs with known answers; edit-op breakdown sums correctly; length-ratio handles empty inputs.
- **Config**: invalid dtype/hardware combinations rejected; missing paths rejected; deep-merge precedence works.
- **Kaggle helpers**: secret fallback when `UserSecretsClient` raises; path resolution `local ↔ /kaggle/input/`.
- **Smoke** (manual, not CI): `--dry-run --max-train-samples 2 --max-steps 1` on CPU completes end-to-end including a stub WandB run.

### What's intentionally NOT in CI
- Full model load (too large for free runners).
- GPU-dependent code paths (no GPU in CI).
- HF Hub upload (network dependency; would require credentials in CI).

### Reproducibility manifest
Per-run JSON containing: full config, python version, key library versions (`torch`, `transformers`, `unsloth`, `peft`, `huggingface_hub`, `wandb`), git commit hash, seed, GPU device name. Uploaded to W&B and committed to the HF Hub repo.

---

## §12. Risks and Open Questions

### Risks
- **Unsloth Qwen3-VL stability** (added late 2025/early 2026): early-version bugs possible. Mitigation: model id is config-driven, can fall back to Qwen2-VL-2B-Instruct (already cached on disk) for iteration if Qwen3-VL is unstable.
- **9-hour Kaggle kernel limit**: tight on P100 with 922 train samples × 3 epochs. Mitigation: wall-clock projection in config validation refuses configs that would exceed; eval is at epoch end only (not mid-epoch).
- **Small val set (231 samples)**: variance on diacritic-CER will be noticeable. Mitigation: report metrics with confidence intervals where feasible; flagged in the eval-output JSON.
- **Frozen vision tower vs unfrozen**: an open empirical question. Default frozen for v1; ablation flag exists.
- **Synthetic instruction phrasing**: a single fixed instruction may overfit. Mitigation: instruction is a config field; paraphrase ablations are a config swap.

### Open questions (deferred)
- **Test set format and source**: user will provide. Pipeline accepts the same line-OCR text-file shape; if a different format arrives, `data/test_loader.py` will be added at that time.
- **Viet-Receipt-VQA usage**: deferred. Possible future use as a multi-task auxiliary, but not part of v1.
- **Inference / serving**: deferred until eval results justify it.

---

## §13. Out-of-Scope Followups (post-v1)

- Augmentation (rotation ±5°, JPEG jitter, brightness ±20%) — only if v1 underperforms on rotated/blurred subsets.
- Multi-GPU (T4 ×2 / L4 ×4) — adds DDP/FSDP complexity; defer until single-GPU baseline is solid.
- Beam search decoding — defer unless greedy hallucinates.
- Viet-Receipt-VQA multi-task training — needs Phase 0 EDA on that dataset first.
- Real-world held-out test integration — pipeline already accepts an arbitrary path.
