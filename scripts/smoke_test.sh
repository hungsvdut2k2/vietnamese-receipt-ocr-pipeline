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
