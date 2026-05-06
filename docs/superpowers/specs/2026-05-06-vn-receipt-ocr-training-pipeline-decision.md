# Decision Note — Vietnamese Receipt OCR Training Pipeline (Qwen3-VL-2B + Unsloth + LoRA)

**Date:** 2026-05-06
**Spec:** [2026-05-06-vn-receipt-ocr-training-pipeline-design.md](./2026-05-06-vn-receipt-ocr-training-pipeline-design.md)

## Context

Phase 0 EDA on MC-OCR is complete and produced `eda_outputs/{schema.json, training_caps.yaml, normalization_rules.yaml}` — these constrain image resolution (target 888 / cap 1388 longest side), max sequence length (203 tokens, P99=185), and numeric/date/currency normalization rules. plan.md framed Phase 1 as Qwen3-VL-{8B,2B} LoRA on a 24 GB consumer GPU, but the user has narrowed Phase 1 to Qwen3-VL-2B only and shifted runtime to Kaggle GPUs (16 GB P100/T4 free tier). The existing project layout uses `uv` and a single `pyproject.toml` (`vietnamese-receipt-ocr`); the training package extends that, not a fresh project.

During brainstorming the user articulated three durable architectural decisions that bound the design: (1) the model emits free-text, not structured JSON, because the comparison baseline VietOCR also emits raw text and a like-for-like comparison requires output-space parity; (2) the task is whole-document transcription with one fixed instruction per sample, not multi-question VQA — superseding the initial "VQA-only" wording; (3) train and val targets are reconstructed by the same procedure (line-OCR text-files grouped by image-prefix, sorted by index, joined with `\n`), so target distribution is identical across splits. Test set will be provided later in the same shape.

byterover prior context contained Phase 0 EDA artifacts but nothing on training; three pending-review curations were created during brainstorming and need user approval.

## Choice

Build `vn_receipt_ocr` as a subpackage under `src/` of the existing repository. Use Unsloth's `FastVisionModel` for Qwen3-VL-2B loading with LoRA (rank 16, attention + MLP, vision tower frozen by default), AdamW 8-bit + cosine LR, 3 epochs default, batch 1 × grad-accum 8 on a single 16 GB GPU. Configuration is Pydantic v2 with composable YAML profiles (GPU profile + data + model + lora + experiment), validated fail-fast at load time including a wall-clock projection check against Kaggle's session limit. WandB is the required tracker (Kaggle Save-and-Run-All has no interactive log tail) with offline+JSONL fallback when secrets are missing. Best LoRA adapters (selected by `eval/diacritic_cer`) are persisted to a private Hugging Face Hub repo per run; `/kaggle/working/` is treated as ephemeral cache only. Distribution is `pip install git+https://...` primary with a zipped wheel as Kaggle dataset fallback for offline runs; both CLI (`python -m vn_receipt_ocr`) and Python API are exposed. Evaluation is text-only and covers `cer`, `diacritic_cer`, `cer_normalized` (primary) plus `wer`, edit-op breakdown, length-ratio histogram, empty-prediction rate, and latency P50/P95 (diagnostic), with greedy decoding and a 10-sample qualitative W&B Table per epoch.

## Alternatives Rejected

- **KIE / structured JSON output**: rejected. VietOCR baseline does not natively emit JSON; like-for-like comparison would require bolting a rule extractor onto VietOCR. Plain-text output keeps the comparison clean. (User-articulated reasoning during Q2.)
- **Multi-question VQA training**: rejected. The user wants the model to extract whole-document content, not answer per-question prompts. (User-articulated during Q3.)
- **MC-OCR `mcocr_train_df.csv` `anno_texts` as training target**: rejected. `anno_texts` only covers field-annotated regions (SELLER/ADDRESS/TIMESTAMP/TOTAL_COST), but the test target reconstructs full transcribed text from line-OCR — using `anno_texts` would create train/test target distribution mismatch.
- **Augmentation in v1**: rejected. plan.md notes >90% of receipts are near-straight, so heavy aug is unnecessary; light aug deferred to ablation if v1 underperforms. (User confirmed Q8c.)
- **Multi-GPU training (T4 ×2 / L4 ×4) as v1 baseline**: rejected. Unsloth's well-supported single-GPU path is more stable; multi-GPU with DDP/FSDP is paid/limited on Kaggle. GPU profile YAML lets multi-GPU be added later via config, no code change. (Captured in Q1 option D.)
- **Viet-Receipt-VQA dataset for v1**: rejected. The dataset isn't downloaded; even if it were, VQA Q&A pairs don't have a natural "whole-document transcription" target field. Deferred until eval results justify the integration cost.
- **Kaggle Output Datasets for checkpoint persistence**: rejected (as primary). HF Hub is faster, has cleaner versioning, and adapters are small (~50–200 MB). Kaggle Datasets remain a documented offline fallback.
- **Beam search decoding for v1 eval**: rejected. Greedy is faster and simpler; add beam search only if v1 hallucinates measurably.

## Invariants Preserved

- **Train/val/test target shape consistency** (architectural invariant): every split must build full-text targets via the same line-OCR-grouped-by-prefix → sort-by-index → join-by-newline procedure. Mixing in `anno_texts` for any split would break this invariant.
- **EDA artifacts are the source of truth for training caps**: `eda_outputs/training_caps.yaml` (target/max resolution, max_seq_length) and `eda_outputs/normalization_rules.yaml` are consumed by the package — not duplicated, not overridden silently. Updating EDA outputs is the only legitimate way to change these.
- **Fail-fast configuration policy** (carried over from EDA pipeline): every input validated at load time before any data or model touches GPU. Hardware/dtype combinations (e.g. BF16 on P100) are rejected, not silently downgraded.
- **Reproducibility seed = 42** (carried over from EDA): single seed value propagated to torch/numpy/random/transformers/Unsloth.
- **VietOCR comparison parity**: the model's output format must remain plain text. Adding any structured-output mode is a future decision that breaks this comparison and requires its own design pass.
- **Secrets are external**: `WANDB_API_KEY`, `HF_TOKEN` come from Kaggle Secrets. Never hardcoded, never logged.
- **Ephemeral-storage assumption for Kaggle**: durable artifacts live on HF Hub or W&B; `/kaggle/working/` is not durable and the pipeline must not assume otherwise.
- **Shared image directory for both splits** (added during implementation): full receipt JPG for prefix `<X>` lives at `<images_dir>/<X>.jpg` for BOTH train and val splits, where `<images_dir>` is the same `train_images/train_images/`. The dataset's sibling `val_images/` directory holds a different, unlabeled MC-OCR set and must not be used for line-OCR training/eval. Configs, notebook, and dataset wiring all enforce this.
- **Response-only loss masking via prefix-token equality** (added during implementation): `QwenVLCollator` asserts that the tokenized user-only prefix is a literal token-prefix of the tokenized full chat sequence before zeroing the first `prefix_len` label positions. If Qwen-VL's chat template ever changes shape, the assertion fails loudly rather than silently corrupting the loss target.
- **PIL file-handle hygiene at boundaries** (added during implementation): `load_image_rgb` opens images under a context manager so the source FD is released as soon as `convert("RGB")` returns the in-memory copy. Used in both training collation and batch prediction; prevents FD exhaustion on long evaluation runs.

## In-flight Refinements

### 2026-05-06 — Unsloth recorded as `[gpu]` optional extra, not main dep
- **Plan assumed:** Step 1 of Task 1 ran `uv add unsloth ...` putting Unsloth in `[project] dependencies`.
- **Turned out:** Unsloth requires CUDA at install time (xformers wheel build). It cannot be installed on macOS / CPU-only machines, so `uv add` fails locally and CI without GPUs breaks.
- **Chose:** Recorded Unsloth as `[project.optional-dependencies] gpu = ["unsloth"]`. Local dev / CI uses `pip install .` (no Unsloth); Kaggle uses `pip install vn-receipt-ocr[gpu]`.
- **Why:** Preserves the dependency declaration for Kaggle while allowing the rest of the package (config, eval metrics, data loaders) to be developed and unit-tested without GPUs. Lazy-imports of `unsloth` already used in `model/loader.py` make the package importable without it. Tasks 26 (Kaggle notebook) and 27 (README) need to reference the `[gpu]` extras in install instructions.

### 2026-05-06 — Train and val share `train_images/train_images/`; `val_images/` is unrelated
- **Plan assumed:** train and val each had their own image directory, with val crops/full-images living under a `val_images/` sibling.
- **Turned out:** the MC-OCR dataset ships full receipt JPGs for BOTH train and val prefixes under `train_images/train_images/`. The `val_images/` directory holds a different unlabeled set that is irrelevant to line-OCR. The val text manifest's prefixes resolve cleanly into `train_images/train_images/`; 0/231 missing.
- **Chose:** Pointed both `data.train_images_dir` and `data.val_images_dir` at `<root>/train_images/train_images` in `configs/data/mcocr_train_val.yaml` and Cell 5 of `notebooks/kaggle_train.ipynb`. Added a comment in both files documenting that `val_images/` must be ignored.
- **Why:** Matches what the dataset actually ships. Preserves the "train and val targets are reconstructed by the same procedure" invariant from brainstorming — including the same on-disk image source.

### 2026-05-06 — `.gitignore data/` was a wildcard that shadowed tracked dirs
- **Plan assumed:** `data/` in `.gitignore` only matched the project-root data download dir.
- **Turned out:** without a leading `/`, the pattern matched `configs/data/`, `src/vn_receipt_ocr/data/`, and `tests/data/` recursively — silently dropped both `__init__.py` markers from the index for months and forced `git add -f` for tracked YAML configs.
- **Chose:** Anchored to root: `/data/` and `/datasets/`. Restored the missing `__init__.py` markers as part of the same commit.
- **Why:** Restores predictable git behavior. The wildcard had been a quiet source of "why won't git add this?" friction for the entire project. Curated as a domain rule via architectural-memory so the same trap doesn't recur.

### 2026-05-06 — Final code review surfaced two real defects
- **Plan assumed:** `load_image_rgb`'s `Image.open(p).convert("RGB")` and `QwenVLCollator`'s prefix-length subtraction were both correct end-to-end.
- **Turned out:** (a) `Image.open` leaks the source FD because `convert()` returns a new image but does not close the file. Over thousands of eval samples with `batch_size=1`, this would exhaust file descriptors. (b) The collator trusted the chat-template prefix invariant without ever checking that the prefix tokens were a literal prefix of the full sequence; if the template's shape changed in a future Transformers/Unsloth release, training loss would silently mask the wrong tokens.
- **Chose:** (a) Wrap `Image.open` in a `with` block; `convert()` runs inside the block and the source FD closes on exit. (b) Add a `torch.equal` assertion in the collator that fails loudly with a descriptive message if the prefix-token equality breaks.
- **Why:** Both are silent-failure modes that would only manifest in production-scale runs. Loud failure is much cheaper than retrospective debugging of a bad checkpoint or a stuck Kaggle session.
