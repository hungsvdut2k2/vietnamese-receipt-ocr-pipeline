# Decision Note — EDA Pipeline Scope & Structure

**Date:** 2026-05-06
**Spec:** [2026-05-06-eda-pipeline-design.md](2026-05-06-eda-pipeline-design.md)

## Context

The original [`plan.md`](../../../plan.md) §2 specified nine EDA sub-sections covering schema, image, text, manual visual quality, manual annotation audit, receipt-type categorization, VQA structure, cross-dataset comparison, and deliverables. The user opened this brainstorm wanting EDA to **drive experiments** rather than be a descriptive report — meaning every analysis should emit a machine-readable artifact that downstream training/eval scripts depend on. No prior byterover context existed for the project (clean repo, only `plan.md` present). Datasets are not yet downloaded: 5CD-AI/Viet-Receipt-VQA on HuggingFace, MC-OCR 2021 on Kaggle (`domixi1989/vietnamese-receipts-mc-ocr-2021`). During brainstorming, the user repeatedly pruned: manual labeling skipped (Q7), section 2.6 dropped (Q8), section 2.5 dropped (post-Q8), and the Pydantic consumption contract deferred until artifact shapes are observed (Q4 pushback).

## Choice

A single monolithic Jupyter notebook (`eda.ipynb`) executes six automated analyses (2.1, 2.2, 2.3, 2.4-automated, 2.7, 2.8) over both datasets. Each section writes one or more YAML/JSON files to `eda_outputs/`: `schema.json`, `training_caps.yaml`, `normalization_rules.yaml`, `augmentation.yaml`, `prompt_templates.yaml`, `training_strategy.yaml`. A §7 summary cell runs sanity checks (artifact existence, value-range bounds, enum validity) and is the entire regression suite for this phase. Reproducibility is enforced by a single `SEED = 42`, fresh-kernel top-down execution, `eda_outputs/` recreation each run, and recording the HF dataset commit hash + SHA-256 of the Kaggle zip.

## Alternatives Rejected

- **Pydantic typed unified loader at EDA time** — deferred; the user wants to discover artifact shapes before committing to types. Will be revisited in Phase 2 when train.py / eval.py are designed.
- **Per-section notebooks (one notebook per analysis) + `src/eda/` package** — rejected (Q5); user chose monolithic for simpler iteration despite my recommendation of the hybrid.
- **Manual labeling workflows (ipywidgets, Label Studio, spreadsheet)** for 2.4 visual quality and 2.6 annotation audit — skipped (Q7); user judged manual work unnecessary at this stage.
- **Section 2.5 receipt-type categorization for stratified splits** — dropped (post-Q8); without manual labels and without an automated classifier worth building, stratification cannot be honest. Consequence: `splits/{train,val,test}.json` is not produced; training generates its own splits later.
- **Section 2.6 annotation audit** — dropped (Q8); no automated substitute for ground-truth verification. Consequence: no `data_quality.yaml`, no eval ceiling. Reintroducible after Phase 1 if results suggest a ceiling problem.
- **Joint-vs-separate decided by ablation** instead of pre-codified in `training_strategy.yaml` — rejected (Q3); user preferred to codify the pre-decision in artifact form. The ablation could still happen in Phase 3 to validate the codified choice.
- **Rotation measurement via `cv2.minAreaRect` or `pytesseract.image_to_osd`** — rejected (§3 §4 of design); fixed `rotation_range_degrees = [-5, 5]` default is more honest than tuning augmentation on noisy measurements. Also avoids a Tesseract dependency.
- **Tests beyond the §7 sanity checks** — rejected; EDA is one-shot. The §7 cell is the regression suite.

## Invariants Preserved

- The [`plan.md`](../../../plan.md) §1 thesis (Vietnamese-specialized receipt understanding via Qwen3-VL with LoRA on existing public datasets, evaluated against baselines, single 24GB GPU) is unchanged. EDA scope cuts only modify what is *measured* before training, not what is *built*.
- The "EDA emits machine-readable artifacts that experiments depend on" framing (Q2-A) is preserved across every cut: even where sections were dropped, the surviving sections each emit a YAML/JSON artifact rather than narrative findings.
- Fail-fast over defensive try/except — preserved as the error policy.
- Reproducibility discipline — single seed, fresh kernel, dataset version pinning, recreated output dir — preserved.
- **Consequence of dropping 2.5**: the [`plan.md`](../../../plan.md) §5 line "Per receipt-type subset (driven by EDA categorization)" is dead under this scope. Eval can still report by-dataset breakdowns. If per-type breakdowns matter for the final defense, they require reintroducing 2.5 (with manual labels or heuristics) as a separate task.

## In-flight Refinements

- **2026-05-06 — VQA dataset deferred:** `5CD-AI/Viet-Receipt-VQA` (cited in original brainstorm) does not exist on the public HF Hub. The viable substitute `5CD-AI/Viet-OCR-VQA-flash2` is `gated=manual` and access is awaiting review. §5 (`prompt_templates.yaml`) and §6 (`training_strategy.yaml`) are deferred to a follow-up cycle once access is granted. EDA proceeds MC-OCR-only; the spec & plan have been amended in lock-step.
- **Subagent execution constraint:** Plan references "Kernel → Restart & Run All" (interactive Jupyter UI). Subagents cannot run UIs; substitute `uv run jupyter nbconvert --to notebook --execute eda.ipynb --inplace --ExecutePreprocessor.timeout=1800` for end-to-end runs and `nbformat` for cell editing.
- **Notebook structure clarification:** Plan's "8 markdown cells" was an off-by-one — the title cell `# Vietnamese Receipt EDA` sits above the 8 section pairs, giving 17 cells total (9 markdown + 8 code).
- **Cache relocation:** Datasets relocated from `~/.cache/{kagglehub,huggingface}/...` into `./datasets/{kagglehub,huggingface}/...` for project-local visibility. `KAGGLEHUB_CACHE` and `HF_HOME` env vars set at the top of §0 to enforce this.
