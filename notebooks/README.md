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
