# EDA Pipeline — Design Spec

**Date:** 2026-05-06
**Phase:** 0 (Exploratory Data Analysis) of the Vietnamese Receipt Understanding project
**Parent plan:** [`plan.md`](../../../plan.md) §2

## 1. Goal & Scope

### Goal

Run a one-shot exploratory analysis over MC-OCR 2021 (Vietnamese Receipt VQA dataset deferred — see decision note) whose outputs are **machine-readable configuration artifacts** that downstream training and evaluation scripts will depend on. EDA is not a report; it is a generator. If an artifact is missing or stale, downstream experiments are not reproducible.

### In scope

Six automated analyses executed in a single Jupyter notebook (`eda.ipynb`), each writing one or more YAML/JSON files to `eda_outputs/`. Dataset acquisition is included as §0 of the notebook.

| Section | Status | Artifact |
|---|---|---|
| 2.1 Schema & field analysis | included | `schema.json` |
| 2.2 Image characteristics | included | `training_caps.yaml` (resolution fields) |
| 2.3 Text characteristics | included | `normalization_rules.yaml`, `training_caps.yaml` (seq length) |
| 2.4 Visual quality | included (automated only) | `augmentation.yaml` |
| 2.5 Receipt type distribution | **dropped** | — |
| 2.6 Annotation quality audit | **dropped** | — |
| 2.7 VQA structure analysis | **deferred (gated dataset access pending)** | — (deferred) |
| 2.8 Cross-dataset comparison | **deferred (gated dataset access pending)** | — (deferred) |

### Out of scope (explicit)

- **VQA dataset deferred.** `5CD-AI/Viet-OCR-VQA-flash2` (the only viable Vietnamese receipt VQA dataset on HF) is gated and our access request is awaiting manual review. §5 (`prompt_templates.yaml`) and §6 (`training_strategy.yaml`) are deferred until access is granted; current EDA runs MC-OCR only.
- **No manual labeling.** All analyses are automated. Categorical visual-quality buckets (wrinkled / phone vs scan / occluded) are not produced.
- **No annotation audit (2.6).** No `data_quality.yaml`, no annotation-error-rate, no eval-ceiling estimate. May be reintroduced after Phase 1 if results suggest a ceiling problem.
- **No receipt-type categorization (2.5).** No stratified splits artifact. Training will produce its own split logic later (likely random or by-dataset). Eval cannot report per-receipt-type breakdowns under this scope; the line in `plan.md` §5 about "Per receipt-type subset" is dead unless reintroduced.
- **No Pydantic loader / consumption contract.** Artifacts are written as plain YAML/JSON. How `train.py` and `eval.py` consume them is a Phase 2 design decision, deferred until we see what shape EDA actually produces.
- **No training, model code, or eval pipeline.** Strictly EDA.
- **No tests beyond the in-notebook sanity checks** (see §4). EDA is a one-shot exploration, not a long-running service.

### Success criterion

After running `eda.ipynb` end-to-end on a fresh kernel, `eda_outputs/` contains every artifact in the in-scope rows of the table above, all populated with values derived from the data (not placeholders), and the §7 summary cell passes every sanity check. A reader can answer, from those files alone: what the unified schema is, what max image resolution and sequence length we are committing to, what augmentation ranges. Cross-dataset comparison and VQA prompt analysis will be re-enabled once 5CD-AI grants access to `Viet-OCR-VQA-flash2`.

## 2. Artifact Contract

Each artifact's purpose, rough shape, and the eventual consumer. Schemas are intentionally loose (no Pydantic yet); this captures *intent*, not field-level commitments.

| Artifact | Purpose | Shape (sketch) | Future consumer |
|---|---|---|---|
| `schema.json` | Unified field inventory across both datasets | `{fields: [{name, type, source_datasets, coverage_rate, format_observations}]}` | Training prompt constructor, per-field eval |
| `training_caps.yaml` | Hard caps fed to Qwen3-VL | `max_resolution`, `target_resolution`, `max_seq_length`, with P50/P95/P99 of source distributions for justification | `train.py` model config, dynamic-resolution preprocessing |
| `normalization_rules.yaml` | Canonical parsers for noisy fields | `numeric: [patterns + canonical rule]`, `date: [formats + ISO target]`, `currency: [suffix tokens + strip rule]` | `eval.py` field-comparison logic |
| `augmentation.yaml` | Augmentation ranges fitted to data (or fixed defaults where measurement is unreliable) | `rotation_range_degrees`, `brightness_range`, `jpeg_quality_range`, `sharpness_range`, each as `[low, high]` | Training data-loader transforms |
| `prompt_templates.yaml` | VQA prompt taxonomy | `{extraction: [...], reasoning: [...], aggregation: [...]}` with templates | Training prompt constructor |
| `training_strategy.yaml` | Joint / sequential / weighted decision | `strategy`, optional `dataset_weights`, `justification` (short string) | Training data-loader weighting |

## 3. Notebook Structure & Per-Section Computation

### Notebook outline

`eda.ipynb`, sequential top-down execution, single kernel:

```
§0  Setup & data acquisition
§1  Schema & field analysis        → schema.json
§2  Image characteristics          → training_caps.yaml (resolution fields)
§3  Text characteristics           → normalization_rules.yaml + training_caps.yaml (seq length)
§4  Visual quality (automated)     → augmentation.yaml
§5  VQA structure analysis         → prompt_templates.yaml  (deferred)
§6  Cross-dataset comparison       → training_strategy.yaml  (deferred)
§7  Summary + sanity checks
```

### Section dependencies

§1–§5 only depend on §0. §6 depends on the outputs of §1–§5 (it compares distributions, so it runs last). §2, §4, §5 are independent of each other and of §3. §3 benefits from §1's schema (knowing which fields are numeric vs date) but is written to gracefully run independently.

### §0 Setup & data acquisition

- Imports, paths, output dir (`eda_outputs/`, recreated from scratch each run so partial state cannot leak between runs).
- `SEED = 42` set at the top, used for all sampling.
- Viet-Receipt-VQA: `datasets.load_dataset("5CD-AI/Viet-Receipt-VQA")`. Record the HF dataset commit hash.
- MC-OCR: `kagglehub.dataset_download("domixi1989/vietnamese-receipts-mc-ocr-2021")`. Record the SHA-256 of the downloaded zip (Kaggle does not expose a clean hash). Requires `KAGGLE_USERNAME` / `KAGGLE_KEY` in env or a Kaggle account interactive login.
- Parser cell that walks the unpacked MC-OCR folder, infers structure (likely `train/test` split with `images/` + an annotation CSV/TSV), and yields `(image_path, parsed_annotation_dict)`. The MC-OCR parser is best-effort at design time; until the archive is unpacked, exact filenames and column names are unknown. The cell will likely need a one-time tweak when first run.
- Build a unified iterator yielding `(dataset_name, example_id, image, annotation)`.

### §1 Schema & field analysis → `schema.json`

Iterate every annotation; enumerate field names; per field compute coverage rate (% of receipts where present), data type (string / number / date / array), and a small sample of format strings. Note mismatches (same field, different name across datasets). Write `{fields: [{name, type, source_datasets, coverage_rate, format_observations}]}`.

### §2 Image characteristics → `training_caps.yaml` (resolution fields)

Open every image with PIL using `Image.open(...).size` (no full decode); record `(width, height, file_size)`. Plot histograms of longest-side, aspect-ratio, file-size. Compute P50 / P95 / P99 of longest-side. Write `training_caps.yaml` with `max_resolution = P95 of longest-side` (capped at 1920 sanity bound), `target_resolution = P50`, and the raw percentiles alongside for justification.

### §3 Text characteristics → `normalization_rules.yaml` + `training_caps.yaml` (seq length)

For every annotation's text content:
- Count Vietnamese diacritic chars (using a defined char set) divided by total chars.
- Build a `Counter` over all characters (catches encoding issues early).
- Tokenize with the Qwen3-VL tokenizer if it can be loaded in the EDA env; otherwise fall back to whitespace tokens with a 1.3× safety multiplier.
- Regex-extract all numeric and date substrings, group by pattern, count occurrences.

Write `normalization_rules.yaml` with the top numeric / date / currency patterns plus a chosen canonical rule per group. Update `training_caps.yaml` with `max_seq_length = P99 token count` plus a small safety margin.

### §4 Visual quality (automated) → `augmentation.yaml`

Sample N images per dataset (`N = min(500, total)`). For each:
- Sharpness via `cv2.Laplacian(gray, cv2.CV_64F).var()`.
- Brightness via mean intensity.
- JPEG quality from EXIF if present, else estimate via `Image.info.get('quality')`.

Compute `[P5, P95]` for each metric. Write `augmentation.yaml` with `brightness_range`, `jpeg_quality_range`, and `sharpness_range` (informational only — augmentation pipelines do not usually take sharpness, but the value is useful as a quality flag).

**Rotation handling:** rotation is **not measured**. Set `rotation_range_degrees = [-5, 5]` as a fixed default, justified in a comment in the YAML: automated rotation detection (`cv2.minAreaRect` on the dominant contour, or `pytesseract.image_to_osd`) is either too noisy on cluttered backgrounds or adds a Tesseract dependency that isn't worth carrying for a value most receipt-fine-tuning recipes default to anyway. We document the choice rather than tune augmentation on noisy measurements.

### §5 VQA structure → `prompt_templates.yaml`

Viet-Receipt-VQA only. For each `(image, [questions])` pair:
- Classify each question with a keyword-rule dispatcher: e.g., `"tổng" / "total" / "sum"` → aggregation; `"có bao nhiêu" / "how many"` → reasoning; default → extraction.
- Count question types, questions-per-image, answer lengths.
- Sample 5–10 representative templates per type.

Write `prompt_templates.yaml` with `{extraction: [...], reasoning: [...], aggregation: [...]}`.

### §6 Cross-dataset comparison → `training_strategy.yaml`

Read the outputs of §1–§3 split per dataset. Quantify distribution overlap on:
- Schema (Jaccard over field names).
- Resolution (KS test on longest-side).
- Text length (KS test).

Apply a rule:
- All three measures show high overlap → `strategy: joint`.
- All three diverge sharply → `strategy: sequential`.
- Mixed → `strategy: weighted` with `dataset_weights` set inversely proportional to size.

Thresholds for "high overlap" / "diverge sharply" are documented in the cell. Write `training_strategy.yaml` with the choice, weights (if applicable), and a 1–2 sentence justification string.

### §7 Summary + sanity checks

Print every artifact path with key values inline. Run the sanity checks listed in §4 below. Fail loud on any violation.

## 4. Reproducibility, Dependencies, Sanity Checks

### Reproducibility

- Single random seed (`SEED = 42`) at the top of the notebook, used for all sampling. Documented in §7.
- Dataset versions pinned: HF dataset commit hash recorded; SHA-256 of the Kaggle zip recorded.
- Notebook executes top-down on a fresh kernel — no out-of-order cell dependencies. §0 recreates `eda_outputs/` from scratch.

### Dependencies (EDA-phase `requirements.txt`)

`datasets`, `kagglehub`, `Pillow`, `opencv-python`, `numpy`, `pandas`, `matplotlib`, `pyyaml`, `transformers` (for the Qwen3-VL tokenizer in §3 — optional fallback to whitespace tokens with a 1.3× safety multiplier if the model is not downloadable in the EDA env).

No Tesseract. No labeling tools.

### Sanity checks (§7 cell)

All checks fail loud with a clear error message naming the offending artifact and value. No try/except; defensive error-handling would hide real problems for a one-shot exploration.

- Every expected artifact path exists.
- `training_caps.yaml`: `max_resolution ∈ [256, 4096]`, `max_seq_length ∈ [128, 8192]`.
- `augmentation.yaml`: every `[low, high]` range satisfies `low < high`.
- `training_strategy.yaml`: `strategy ∈ {joint, sequential, weighted}`.
- `schema.json`: `fields` is non-empty.
- `prompt_templates.yaml`: each category has ≥ 1 template.

The §7 cell is the entire regression check — there is no separate pytest suite for EDA.

### Error policy beyond sanity checks

Fail-fast everywhere. If a download fails, an annotation is malformed, an image is unreadable, the notebook aborts at that cell with the underlying exception. Transient errors do not exist in this context.
