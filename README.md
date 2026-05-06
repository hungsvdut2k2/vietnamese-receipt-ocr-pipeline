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
