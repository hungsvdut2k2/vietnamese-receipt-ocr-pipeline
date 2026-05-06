# EDA Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a single-notebook EDA pipeline (`eda.ipynb`) that produces configuration artifacts in `eda_outputs/` from MC-OCR 2021 (Vietnamese Receipt VQA dataset deferred — see decision note), with all sanity checks living in the notebook's §7 cell.

**Architecture:** One monolithic Jupyter notebook with eight sections (§0 setup + dataset acquisition, §1–§6 analyses, §7 summary + sanity checks). Each analysis section emits one or more YAML/JSON artifacts to `eda_outputs/`. The §7 cell holds all sanity-check assertions and is the entire regression suite — no separate pytest. Reproducibility via single seed, fresh-kernel top-down execution, dataset version pinning.

**Tech Stack:** Python 3.10+, **uv** (package manager), Jupyter, `datasets` (HF), `kagglehub`, Pillow, OpenCV, NumPy, pandas, Matplotlib, PyYAML, transformers (Qwen3-VL tokenizer with whitespace fallback), scipy.

**Spec:** [docs/superpowers/specs/2026-05-06-eda-pipeline-design.md](../specs/2026-05-06-eda-pipeline-design.md)
**Decision note:** [docs/superpowers/specs/2026-05-06-eda-pipeline-decision.md](../specs/2026-05-06-eda-pipeline-decision.md)

---

## Scope Amendment (2026-05-06): VQA Deferred

The original plan referenced `5CD-AI/Viet-Receipt-VQA` which does not exist on the public HF Hub. The viable substitute `5CD-AI/Viet-OCR-VQA-flash2` is gated; access is awaiting manual review. The user has elected to proceed MC-OCR-only for the current cycle.

**Tasks 7 (§5 VQA Structure) and 8 (§6 Cross-Dataset Comparison) are deferred** until VQA access is granted. The other tasks proceed with MC-OCR alone. When VQA access is granted, Tasks 7 and 8 can be re-enabled without re-doing the earlier sections (each section's MC-OCR-only logic remains valid; the deferred tasks add cross-dataset analysis on top).

---

## File Structure

To create:
- `pyproject.toml` — project metadata + dependencies (managed by `uv`)
- `uv.lock` — uv-generated lockfile (committed for reproducibility)
- `.python-version` — uv-generated pinned Python version
- `.gitignore` — exclude generated artifacts and venv
- `eda.ipynb` — the notebook (single file containing all 8 sections)
- `eda_outputs/` — runtime-generated artifacts directory (gitignored)

To modify: none.

Each task in this plan adds one section's cells to `eda.ipynb` plus the matching `validate_*` function to the §7 cell. The TDD pattern per task is:
1. Extend §7 with a `validate_section_N` that asserts the new artifact's existence and basic validity.
2. Restart kernel, Run All — confirm §7 fails on the new assertion.
3. Implement the §N cell so the artifact is produced.
4. Restart kernel, Run All — confirm §7 passes.
5. Commit.

"Restart kernel, Run All" means: in Jupyter, `Kernel → Restart & Run All`. This guarantees no stale variables leak between runs and that artifacts are regenerated from scratch.

---

## Task 1: Project Scaffolding (uv)

**Files:**
- Create: `pyproject.toml`, `uv.lock`, `.python-version` (via `uv`)
- Create: `.gitignore`
- Create: `eda.ipynb` (with section markdown headers, empty code cells)

- [ ] **Step 1: Initialize a uv project**

Run from the project root:

```bash
uv init --no-readme --no-package --python 3.11
```

Expected: creates `pyproject.toml`, `.python-version`, and a stub `main.py` (we'll delete it shortly).

Delete the stub:

```bash
rm -f main.py hello.py
```

- [ ] **Step 2: Add dependencies via `uv add`**

```bash
uv add datasets kagglehub Pillow opencv-python numpy pandas matplotlib PyYAML transformers ipykernel notebook scipy
```

Expected: `uv` resolves and locks the dependency graph, writing `uv.lock` and updating `pyproject.toml` with a `[project] dependencies` block. A `.venv/` directory is created automatically.

Verify:

```bash
uv run python -c "import datasets, kagglehub, cv2, yaml, transformers; print('ok')"
```

Expected: `ok`.

- [ ] **Step 3: Register the venv as a Jupyter kernel**

```bash
uv run python -m ipykernel install --user --name vietnamese-receipt-eda --display-name "Python (vietnamese-receipt-eda)"
```

Expected: `Installed kernelspec vietnamese-receipt-eda in ...`.

- [ ] **Step 4: Create `.gitignore`**

```
# Python
__pycache__/
*.py[cod]
.venv/
venv/

# uv
# (uv.lock IS committed; .python-version IS committed)

# Jupyter
.ipynb_checkpoints/

# EDA artifacts (regenerated from notebook)
eda_outputs/

# Datasets (downloaded at runtime)
data/
*.zip

# OS
.DS_Store
```

- [ ] **Step 5: Create `eda.ipynb` with section scaffolding**

Launch Jupyter:

```bash
uv run jupyter notebook
```

Create `eda.ipynb` and add 8 markdown cells, each followed by an empty code cell. Markdown contents (one per cell):

```
# Vietnamese Receipt EDA
```

```
## §0 Setup & Data Acquisition
```

```
## §1 Schema & Field Analysis → schema.json
```

```
## §2 Image Characteristics → training_caps.yaml (resolution)
```

```
## §3 Text Characteristics → normalization_rules.yaml + training_caps.yaml (seq_length)
```

```
## §4 Visual Quality (Automated) → augmentation.yaml
```

```
## §5 VQA Structure → prompt_templates.yaml
```

```
## §6 Cross-Dataset Comparison → training_strategy.yaml
```

```
## §7 Summary & Sanity Checks
```

Set the kernel to "Python (vietnamese-receipt-eda)" via `Kernel → Change kernel`. Save.

- [ ] **Step 6: Commit**

```bash
git init  # only if repo not yet initialized
git add pyproject.toml uv.lock .python-version .gitignore eda.ipynb
git commit -m "chore: scaffold EDA notebook and uv project"
```

---

## Task 2: §0 Setup & Data Acquisition

**Files:**
- Modify: `eda.ipynb` (§0 code cell, §7 code cell — first version)

- [ ] **Step 1: Write the §7 sanity-check stub**

Paste into the §7 code cell:

```python
import os, json, yaml
from pathlib import Path

OUT = Path("eda_outputs")

def validate_section_0():
    """§0 produced MC-OCR iterator with non-zero count."""
    assert "MCOCR_COUNT" in globals(), "§0 did not run; MCOCR_COUNT undefined"
    assert MCOCR_COUNT > 0, "MC-OCR yielded zero examples"
    print(f"  §0 OK — MC-OCR: {MCOCR_COUNT}")

def run_all_validations():
    print("Running sanity checks...")
    validate_section_0()
    print("All sanity checks passed.")

run_all_validations()
```

- [ ] **Step 2: Run §7, confirm it fails**

Using `nbconvert`: `uv run jupyter nbconvert --to notebook --execute eda.ipynb --inplace --ExecutePreprocessor.timeout=1800`.
Expected: §7 raises `AssertionError: §0 did not run; MCOCR_COUNT undefined`.

- [ ] **Step 3: Write §0 cell**

Paste into the §0 code cell:

```python
import os
import random
import hashlib
import json
import yaml
import shutil
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
from PIL import Image

# --- Project-local cache dirs (gitignored) ---
PROJECT_ROOT = Path(".").resolve()
DATASETS_DIR = PROJECT_ROOT / "datasets"
os.environ["KAGGLEHUB_CACHE"] = str(DATASETS_DIR / "kagglehub")
os.environ["HF_HOME"] = str(DATASETS_DIR / "huggingface")
(DATASETS_DIR / "kagglehub").mkdir(parents=True, exist_ok=True)
(DATASETS_DIR / "huggingface").mkdir(parents=True, exist_ok=True)

# --- Reproducibility ---
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# --- Output dir (recreate from scratch each run) ---
OUT = Path("eda_outputs")
if OUT.exists():
    shutil.rmtree(OUT)
OUT.mkdir(parents=True)

# --- Download MC-OCR from Kaggle ---
# NOTE: VQA (`5CD-AI/Viet-OCR-VQA-flash2`) is deferred — gated access pending manual review.
import kagglehub
print("Downloading MC-OCR from Kaggle (cached under ./datasets/kagglehub/)...")
mcocr_path = Path(kagglehub.dataset_download("domixi1989/vietnamese-receipts-mc-ocr-2021"))
print(f"MC-OCR unpacked to: {mcocr_path}")

# Reproducibility hash for MC-OCR (Kaggle has no clean revision)
def tree_hash(path):
    h = hashlib.sha256()
    for p in sorted(path.rglob("*")):
        if p.is_file():
            h.update(str(p.relative_to(path)).encode())
            h.update(str(p.stat().st_size).encode())
    return h.hexdigest()
MCOCR_TREE_HASH = tree_hash(mcocr_path)
print(f"MC-OCR tree hash (size-based): {MCOCR_TREE_HASH[:16]}...")

# --- MC-OCR layout ---
# Confirmed via prior inspection of versions/17/ archive:
#   CSV:        mcocr_train_df.csv at the archive root
#   Image dir:  data0.7/data0.7/
#   Filename:   img_id column
mcocr_csv = mcocr_path / "mcocr_train_df.csv"
mcocr_img_dir = mcocr_path / "data0.7" / "data0.7"
print(f"MC-OCR CSV: {mcocr_csv}")
print(f"MC-OCR image dir: {mcocr_img_dir}")
mcocr_df = pd.read_csv(mcocr_csv)
print(f"MC-OCR CSV columns: {list(mcocr_df.columns)}")
print(f"MC-OCR CSV head:\n{mcocr_df.head(2).to_string()}")

def iter_mcocr():
    """Yield (example_id, image_path_str, annotation_dict)."""
    for _, row in mcocr_df.iterrows():
        name = str(row["img_id"])
        img_path = mcocr_img_dir / name
        if not img_path.exists():
            continue
        yield (name, str(img_path), row.to_dict())

# --- Materialize counts (cheap) ---
MCOCR_COUNT = sum(1 for _ in iter_mcocr())
print(f"\nMC-OCR examples reachable: {MCOCR_COUNT}")
```

- [ ] **Step 4: Execute & verify §7 passes**

Run: `uv run jupyter nbconvert --to notebook --execute eda.ipynb --inplace --ExecutePreprocessor.timeout=1800`.
Expected: §0 prints MC-OCR path and count; §7 prints `§0 OK — MC-OCR: <N>` and `All sanity checks passed.`

- [ ] **Step 5: Strip notebook outputs & commit**

Run: `uv run jupyter nbconvert --clear-output --inplace eda.ipynb`.
Then:

```bash
git add eda.ipynb docs/superpowers/specs/2026-05-06-eda-pipeline-design.md docs/superpowers/specs/2026-05-06-eda-pipeline-decision.md docs/superpowers/plans/2026-05-06-eda-pipeline.md
git commit -m "feat(eda): §0 MC-OCR setup; defer VQA-dependent §5/§6 pending dataset access"
```

---

## Task 3: §1 Schema & Field Analysis → schema.json

**Files:**
- Modify: `eda.ipynb` (§1 code cell + §7 cell extension)

- [ ] **Step 1: Extend §7 with `validate_section_1`**

In the §7 cell, ABOVE `def run_all_validations()`, insert:

```python
def validate_section_1():
    """§1 emits schema.json with non-empty fields list and valid coverage rates."""
    p = OUT / "schema.json"
    assert p.exists(), f"{p} missing"
    schema = json.loads(p.read_text(encoding="utf-8"))
    assert "fields" in schema, "schema.json has no 'fields' key"
    assert len(schema["fields"]) > 0, "schema.json fields list is empty"
    for f in schema["fields"]:
        for key in ("name", "type", "source_datasets", "coverage_rate", "format_observations"):
            assert key in f, f"field missing key '{key}': {f}"
        assert 0.0 <= f["coverage_rate"] <= 1.0, f"coverage_rate out of range: {f}"
        assert f["type"] in ("string", "number", "date", "array"), f"unknown type: {f}"
    print(f"  §1 OK — {len(schema['fields'])} fields cataloged")
```

In `run_all_validations`, append `validate_section_1()` after `validate_section_0()`.

- [ ] **Step 2: Restart & Run All, confirm §7 fails**

Expected: `AssertionError: eda_outputs/schema.json missing`.

- [ ] **Step 3: Write §1 cell**

Paste into the §1 code cell:

```python
def annotation_pairs():
    """Yield (dataset_name, ann_dict) for both datasets."""
    for _id, _img, ann in iter_mcocr():
        yield ("mc_ocr", ann if isinstance(ann, dict) else dict(ann))
    for _id, _img, ann in iter_vqa():
        yield ("vqa", ann if isinstance(ann, dict) else dict(ann))

field_obs = {}
totals = Counter()

for ds, ann in annotation_pairs():
    totals[ds] += 1
    for k, v in ann.items():
        rec = field_obs.setdefault(k, {"datasets": set(), "present_per_ds": Counter(), "samples": []})
        rec["datasets"].add(ds)
        if v is not None and v != "" and v != [] and not (isinstance(v, float) and pd.isna(v)):
            rec["present_per_ds"][ds] += 1
            if len(rec["samples"]) < 5:
                rec["samples"].append(str(v)[:120])

def infer_type(samples):
    if not samples:
        return "string"
    if all(s.replace(",", "").replace(".", "").replace("-", "").replace(" ", "").isdigit()
           for s in samples if s):
        return "number"
    if any(("/" in s and len(s) <= 12) or ("-" in s and len(s) == 10) for s in samples):
        return "date"
    if any(s.lstrip().startswith("[") for s in samples):
        return "array"
    return "string"

fields = []
for name, rec in sorted(field_obs.items()):
    denom = sum(totals[d] for d in rec["datasets"])
    coverage = sum(rec["present_per_ds"].values()) / denom if denom else 0.0
    fields.append({
        "name": name,
        "type": infer_type(rec["samples"]),
        "source_datasets": sorted(rec["datasets"]),
        "coverage_rate": round(coverage, 3),
        "format_observations": rec["samples"][:5],
    })

schema = {"fields": fields, "totals": dict(totals)}
(OUT / "schema.json").write_text(json.dumps(schema, indent=2, ensure_ascii=False), encoding="utf-8")
print(f"§1 wrote schema.json with {len(fields)} fields")
for f in fields[:10]:
    print(f"  {f['name']:30s} type={f['type']:7s} cov={f['coverage_rate']:.2f} ds={f['source_datasets']}")
```

- [ ] **Step 4: Restart & Run All, verify §7 passes**

Expected: §1 prints first 10 fields; §7 prints `§1 OK — N fields cataloged`.

- [ ] **Step 5: Commit**

```bash
git add eda.ipynb
git commit -m "feat(eda): §1 schema and field analysis"
```

---

## Task 4: §2 Image Characteristics → training_caps.yaml (resolution fields)

**Files:**
- Modify: `eda.ipynb` (§2 code cell + §7 cell extension)

- [ ] **Step 1: Extend §7 with `validate_section_2`**

Insert in §7:

```python
def validate_section_2():
    """§2 emits training_caps.yaml with valid max_resolution and percentiles."""
    p = OUT / "training_caps.yaml"
    assert p.exists(), f"{p} missing"
    caps = yaml.safe_load(p.read_text(encoding="utf-8"))
    for key in ("max_resolution", "target_resolution", "longest_side_percentiles"):
        assert key in caps, f"training_caps.yaml missing key '{key}'"
    assert 256 <= caps["max_resolution"] <= 4096, f"max_resolution {caps['max_resolution']} out of [256, 4096]"
    assert 256 <= caps["target_resolution"] <= caps["max_resolution"], \
        f"target_resolution {caps['target_resolution']} not in [256, max_resolution]"
    assert "p50" in caps["longest_side_percentiles"], "percentiles missing p50"
    print(f"  §2 OK — max_resolution={caps['max_resolution']}, target={caps['target_resolution']}")
```

Append `validate_section_2()` to `run_all_validations`.

- [ ] **Step 2: Restart & Run All, confirm §7 fails**

Expected: `AssertionError: eda_outputs/training_caps.yaml missing`.

- [ ] **Step 3: Write §2 cell**

```python
import matplotlib.pyplot as plt

def open_size(img_or_path):
    if hasattr(img_or_path, "size") and not isinstance(img_or_path, (str, Path)):
        return img_or_path.size  # PIL image -> (W, H)
    return Image.open(img_or_path).size

records = []
for _id, img_path, _ann in iter_mcocr():
    try:
        w, h = open_size(img_path)
        sz = Path(img_path).stat().st_size
        records.append({"dataset": "mc_ocr", "w": w, "h": h, "file_size": sz})
    except Exception:
        continue
for _id, img, _ann in iter_vqa():
    try:
        w, h = open_size(img)
        records.append({"dataset": "vqa", "w": w, "h": h, "file_size": None})
    except Exception:
        continue

df_img = pd.DataFrame(records)
df_img["longest_side"] = df_img[["w", "h"]].max(axis=1)
df_img["aspect_ratio"] = df_img["h"] / df_img["w"]

print(f"§2 image stats over {len(df_img)} images:")
print(df_img.groupby("dataset")[["longest_side", "aspect_ratio"]].describe().to_string())

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
df_img["longest_side"].hist(bins=50, ax=axes[0]); axes[0].set_title("Longest side (px)")
df_img["aspect_ratio"].hist(bins=50, ax=axes[1]); axes[1].set_title("Aspect ratio (h/w)")
df_img["file_size"].dropna().hist(bins=50, ax=axes[2]); axes[2].set_title("File size (bytes)")
plt.tight_layout()
plt.show()

ls = df_img["longest_side"]
percentiles = {"p50": int(ls.quantile(0.50)), "p95": int(ls.quantile(0.95)), "p99": int(ls.quantile(0.99))}
max_res = min(1920, percentiles["p95"])
target_res = percentiles["p50"]

caps = {
    "max_resolution": int(max_res),
    "target_resolution": int(target_res),
    "longest_side_percentiles": percentiles,
    "n_images_analyzed": int(len(df_img)),
    "max_resolution_rule": "min(P95 of longest-side, 1920) — 1920 is the sanity bound for VRAM",
}
(OUT / "training_caps.yaml").write_text(yaml.safe_dump(caps, sort_keys=False), encoding="utf-8")
print(f"§2 wrote training_caps.yaml: max_resolution={max_res}, target_resolution={target_res}")
```

- [ ] **Step 4: Restart & Run All, verify §7 passes**

Expected: §2 prints stats + plots; §7 prints `§2 OK — max_resolution=..., target=...`.

- [ ] **Step 5: Commit**

```bash
git add eda.ipynb
git commit -m "feat(eda): §2 image characteristics and resolution caps"
```

---

## Task 5: §3 Text Characteristics → normalization_rules.yaml + max_seq_length

**Files:**
- Modify: `eda.ipynb` (§3 code cell + §7 cell extension)

- [ ] **Step 1: Extend §7 with `validate_section_3`**

Insert in §7:

```python
def validate_section_3():
    """§3 emits normalization_rules.yaml + appends max_seq_length to training_caps.yaml."""
    p_norm = OUT / "normalization_rules.yaml"
    p_caps = OUT / "training_caps.yaml"
    assert p_norm.exists(), f"{p_norm} missing"
    norm = yaml.safe_load(p_norm.read_text(encoding="utf-8"))
    for key in ("numeric", "date", "currency"):
        assert key in norm, f"normalization_rules.yaml missing '{key}'"
        assert "patterns" in norm[key] and "canonical" in norm[key], \
            f"normalization_rules.yaml '{key}' missing patterns/canonical"

    caps = yaml.safe_load(p_caps.read_text(encoding="utf-8"))
    assert "max_seq_length" in caps, "training_caps.yaml missing max_seq_length"
    assert 128 <= caps["max_seq_length"] <= 8192, f"max_seq_length out of [128, 8192]: {caps['max_seq_length']}"
    print(f"  §3 OK — max_seq_length={caps['max_seq_length']}, "
          f"{len(norm['numeric']['patterns'])} numeric patterns, "
          f"{len(norm['date']['patterns'])} date patterns")
```

Append `validate_section_3()` to `run_all_validations`.

- [ ] **Step 2: Restart & Run All, confirm §7 fails**

Expected: `AssertionError: eda_outputs/normalization_rules.yaml missing`.

- [ ] **Step 3: Write §3 cell**

```python
import re

VN_DIACRITIC_CHARS = set(
    "àáảãạăằắẳẵặâầấẩẫậ"
    "èéẻẽẹêềếểễệ"
    "ìíỉĩị"
    "òóỏõọôồốổỗộơờớởỡợ"
    "ùúủũụưừứửữự"
    "ỳýỷỹỵ"
    "đ"
    "ÀÁẢÃẠĂẰẮẲẴẶÂẦẤẨẪẬ"
    "ÈÉẺẼẸÊỀẾỂỄỆ"
    "ÌÍỈĨỊ"
    "ÒÓỎÕỌÔỒỐỔỖỘƠỜỚỞỠỢ"
    "ÙÚỦŨỤƯỪỨỬỮỰ"
    "ỲÝỶỸỴ"
    "Đ"
)

def text_payloads():
    """Yield strings extracted from annotations of both datasets."""
    for ds, ann in annotation_pairs():
        for k, v in ann.items():
            if isinstance(v, str) and v:
                yield v
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, str) and item:
                        yield item

char_counter = Counter()
diac_chars = 0
total_chars = 0
for s in text_payloads():
    char_counter.update(s)
    diac_chars += sum(1 for c in s if c in VN_DIACRITIC_CHARS)
    total_chars += len(s)
diac_density = diac_chars / total_chars if total_chars else 0.0
print(f"§3 char stats: total_chars={total_chars}, diacritic_density={diac_density:.4f}, "
      f"unique_chars={len(char_counter)}")

NUMERIC_RE = re.compile(r"\d{1,3}(?:[.,\s]\d{3})*(?:[.,]\d+)?")
DATE_RE = re.compile(r"\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b")
CURRENCY_RE = re.compile(r"(đ|₫|VND|VNĐ|đồng)\b", re.IGNORECASE)

def pattern_signature(s):
    return "".join("D" if c.isdigit() else c for c in s)

numeric_patterns = Counter()
for s in text_payloads():
    for m in NUMERIC_RE.findall(s):
        numeric_patterns[pattern_signature(m)] += 1

date_patterns = Counter()
for s in text_payloads():
    for m in DATE_RE.findall(s):
        date_patterns[pattern_signature(m)] += 1

currency_tokens = Counter()
for s in text_payloads():
    for m in CURRENCY_RE.findall(s):
        currency_tokens[m.lower()] += 1

print("Top numeric patterns:", numeric_patterns.most_common(10))
print("Top date patterns:   ", date_patterns.most_common(10))
print("Currency tokens:     ", currency_tokens.most_common())

def get_tokenizer():
    try:
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained("Qwen/Qwen2-VL-2B-Instruct"), 1.0
    except Exception as e:
        print(f"Tokenizer unavailable ({e!r}); using whitespace token count × 1.3 safety multiplier")
        return None, 1.3

tok, safety = get_tokenizer()

def count_tokens(s):
    if tok is not None:
        return len(tok.encode(s, add_special_tokens=False))
    return int(len(s.split()) * safety)

def receipt_text(ann):
    parts = []
    for v in ann.values():
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, list):
            for item in v:
                if isinstance(item, str):
                    parts.append(item)
    return "\n".join(parts)

token_counts = []
for _ds, ann in annotation_pairs():
    token_counts.append(count_tokens(receipt_text(ann)))
tc = pd.Series(token_counts)
p99 = int(tc.quantile(0.99))
max_seq_length = max(128, min(int(p99 * 1.1), 8192))
print(f"§3 token-count percentiles: p50={int(tc.quantile(0.5))}, "
      f"p95={int(tc.quantile(0.95))}, p99={p99}; max_seq_length={max_seq_length}")

norm = {
    "numeric": {
        "patterns": [{"pattern": p, "count": c} for p, c in numeric_patterns.most_common(20)],
        "canonical": "Strip thousands separators (. , space). Replace decimal comma with dot. Parse as float.",
    },
    "date": {
        "patterns": [{"pattern": p, "count": c} for p, c in date_patterns.most_common(20)],
        "canonical": "Parse to ISO 8601 (YYYY-MM-DD). Heuristic: if first part > 31, assume YYYY-MM-DD; else DD/MM/YYYY.",
    },
    "currency": {
        "patterns": [{"token": t, "count": c} for t, c in currency_tokens.most_common()],
        "canonical": "Strip currency suffix tokens (đ, ₫, VND, VNĐ, đồng) before numeric parse.",
    },
    "diacritic_density": round(diac_density, 4),
}
(OUT / "normalization_rules.yaml").write_text(
    yaml.safe_dump(norm, sort_keys=False, allow_unicode=True), encoding="utf-8"
)
print(f"§3 wrote normalization_rules.yaml")

caps_path = OUT / "training_caps.yaml"
caps = yaml.safe_load(caps_path.read_text(encoding="utf-8"))
caps["max_seq_length"] = max_seq_length
caps["token_count_percentiles"] = {
    "p50": int(tc.quantile(0.5)),
    "p95": int(tc.quantile(0.95)),
    "p99": p99,
}
caps_path.write_text(yaml.safe_dump(caps, sort_keys=False), encoding="utf-8")
print(f"§3 updated training_caps.yaml with max_seq_length={max_seq_length}")
```

- [ ] **Step 4: Restart & Run All, verify §7 passes**

Expected: §3 prints char stats, top patterns, token percentiles; §7 prints `§3 OK — max_seq_length=..., N numeric patterns, M date patterns`.

- [ ] **Step 5: Commit**

```bash
git add eda.ipynb
git commit -m "feat(eda): §3 text characteristics and normalization rules"
```

---

## Task 6: §4 Visual Quality (Automated) → augmentation.yaml

**Files:**
- Modify: `eda.ipynb` (§4 code cell + §7 cell extension)

- [ ] **Step 1: Extend §7 with `validate_section_4`**

Insert in §7:

```python
def validate_section_4():
    """§4 emits augmentation.yaml with valid ranges and fixed rotation default."""
    p = OUT / "augmentation.yaml"
    assert p.exists(), f"{p} missing"
    aug = yaml.safe_load(p.read_text(encoding="utf-8"))
    for key in ("rotation_range_degrees", "brightness_range", "jpeg_quality_range", "sharpness_range"):
        assert key in aug, f"augmentation.yaml missing '{key}'"
        rng = aug[key]
        assert isinstance(rng, list) and len(rng) == 2, f"{key} not a [low, high] list: {rng}"
        assert rng[0] < rng[1], f"{key} has low >= high: {rng}"
    assert aug["rotation_range_degrees"] == [-5, 5], \
        f"rotation_range_degrees must be fixed [-5, 5] per spec: {aug['rotation_range_degrees']}"
    print(f"  §4 OK — augmentation ranges valid; rotation fixed at [-5, 5]")
```

Append `validate_section_4()` to `run_all_validations`.

- [ ] **Step 2: Restart & Run All, confirm §7 fails**

Expected: `AssertionError: eda_outputs/augmentation.yaml missing`.

- [ ] **Step 3: Write §4 cell**

```python
import cv2

N_SAMPLE = 500

def sample_paths_mcocr(n):
    paths = [p for _id, p, _ann in iter_mcocr()]
    random.shuffle(paths)
    return paths[:n]

def sample_pils_vqa(n):
    out = []
    for _id, img, _ann in iter_vqa():
        out.append(img)
        if len(out) >= n:
            break
    return out

def to_cv2(img_or_path):
    if isinstance(img_or_path, (str, Path)):
        return cv2.imdecode(np.fromfile(str(img_or_path), dtype=np.uint8), cv2.IMREAD_COLOR)
    arr = np.array(img_or_path.convert("RGB"))
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

def visual_metrics(cv_img):
    gray = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
    sharp = cv2.Laplacian(gray, cv2.CV_64F).var()
    brightness = float(gray.mean())
    return sharp, brightness

def jpeg_quality_estimate(img_or_path):
    try:
        if isinstance(img_or_path, (str, Path)):
            with Image.open(img_or_path) as im:
                return im.info.get("quality")
        return img_or_path.info.get("quality")
    except Exception:
        return None

samples = []
for p in sample_paths_mcocr(min(N_SAMPLE, MCOCR_COUNT)):
    cv_img = to_cv2(p)
    if cv_img is None:
        continue
    sharp, brightness = visual_metrics(cv_img)
    samples.append({"dataset": "mc_ocr", "sharpness": sharp, "brightness": brightness,
                    "jpeg_quality": jpeg_quality_estimate(p)})
for img in sample_pils_vqa(min(N_SAMPLE, VQA_COUNT)):
    cv_img = to_cv2(img)
    if cv_img is None:
        continue
    sharp, brightness = visual_metrics(cv_img)
    samples.append({"dataset": "vqa", "sharpness": sharp, "brightness": brightness,
                    "jpeg_quality": jpeg_quality_estimate(img)})

df_vis = pd.DataFrame(samples)
print(f"§4 visual metrics over {len(df_vis)} images:")
print(df_vis.groupby("dataset")[["sharpness", "brightness"]].describe().to_string())

def rng_p5_p95(series):
    s = series.dropna()
    if len(s) == 0:
        return None
    return [float(s.quantile(0.05)), float(s.quantile(0.95))]

aug = {
    "rotation_range_degrees": [-5, 5],
    "rotation_note": "Fixed default; not data-derived. Automated rotation detection was rejected as either too noisy on cluttered backgrounds or requiring a Tesseract dependency. See decision note 2026-05-06-eda-pipeline-decision.md.",
    "brightness_range": rng_p5_p95(df_vis["brightness"]) or [80.0, 200.0],
    "jpeg_quality_range": rng_p5_p95(df_vis["jpeg_quality"]) or [70, 95],
    "sharpness_range": rng_p5_p95(df_vis["sharpness"]) or [50.0, 1500.0],
    "sharpness_note": "Informational only; augmentation pipelines do not typically take sharpness ranges as input.",
    "n_images_sampled": int(len(df_vis)),
}
(OUT / "augmentation.yaml").write_text(
    yaml.safe_dump(aug, sort_keys=False, allow_unicode=True), encoding="utf-8"
)
print(f"§4 wrote augmentation.yaml: rotation={aug['rotation_range_degrees']}, "
      f"brightness={aug['brightness_range']}, jpeg_quality={aug['jpeg_quality_range']}")
```

- [ ] **Step 4: Restart & Run All, verify §7 passes**

Expected: §4 prints visual stats; §7 prints `§4 OK — augmentation ranges valid; rotation fixed at [-5, 5]`.

- [ ] **Step 5: Commit**

```bash
git add eda.ipynb
git commit -m "feat(eda): §4 automated visual quality and augmentation ranges"
```

---

## Task 7: §5 VQA Structure → prompt_templates.yaml

**[DEFERRED — pending VQA access]**

**Files:**
- Modify: `eda.ipynb` (§5 code cell + §7 cell extension)

- [ ] **Step 1: Extend §7 with `validate_section_5`**

Insert in §7:

```python
def validate_section_5():
    """§5 emits prompt_templates.yaml with all three categories non-empty."""
    p = OUT / "prompt_templates.yaml"
    assert p.exists(), f"{p} missing"
    pt = yaml.safe_load(p.read_text(encoding="utf-8"))
    for key in ("extraction", "reasoning", "aggregation"):
        assert key in pt, f"prompt_templates.yaml missing '{key}'"
        assert isinstance(pt[key], list) and len(pt[key]) >= 1, f"'{key}' must have ≥ 1 template"
    print(f"  §5 OK — extraction={len(pt['extraction'])}, "
          f"reasoning={len(pt['reasoning'])}, aggregation={len(pt['aggregation'])}")
```

Append `validate_section_5()` to `run_all_validations`.

- [ ] **Step 2: Restart & Run All, confirm §7 fails**

Expected: `AssertionError: eda_outputs/prompt_templates.yaml missing`.

- [ ] **Step 3: Write §5 cell**

```python
AGGREGATION_KEYWORDS = ("tổng", "total", "sum", "cộng", "thành tiền")
REASONING_KEYWORDS = ("bao nhiêu", "how many", "có mấy", "số lượng")

def classify_question(q):
    ql = q.lower()
    if any(k in ql for k in AGGREGATION_KEYWORDS):
        return "aggregation"
    if any(k in ql for k in REASONING_KEYWORDS):
        return "reasoning"
    return "extraction"

per_image_counts = []
type_counter = Counter()
samples_by_type = {"extraction": [], "reasoning": [], "aggregation": []}
answer_lengths = []

for _id, _img, ann in iter_vqa():
    questions = ann.get("questions") or ann.get("question")
    answers = ann.get("answers") or ann.get("answer")
    if isinstance(questions, list):
        per_image_counts.append(len(questions))
        ans_list = answers if isinstance(answers, list) else [None] * len(questions)
        for q, a in zip(questions, ans_list):
            qtype = classify_question(str(q))
            type_counter[qtype] += 1
            if len(samples_by_type[qtype]) < 10:
                samples_by_type[qtype].append(str(q))
            if a is not None:
                answer_lengths.append(len(str(a)))
    elif isinstance(questions, str):
        per_image_counts.append(1)
        qtype = classify_question(questions)
        type_counter[qtype] += 1
        if len(samples_by_type[qtype]) < 10:
            samples_by_type[qtype].append(questions)
        if answers is not None:
            answer_lengths.append(len(str(answers)))

ans_series = pd.Series(answer_lengths) if answer_lengths else pd.Series([0])
qpi_series = pd.Series(per_image_counts) if per_image_counts else pd.Series([0])
print(f"§5 question types: {dict(type_counter)}")
print(f"§5 questions/image: p50={int(qpi_series.quantile(0.5))}, p95={int(qpi_series.quantile(0.95))}")
print(f"§5 answer length: p50={int(ans_series.quantile(0.5))}, p95={int(ans_series.quantile(0.95))}")

def ensure_templates(samples, fallback):
    return samples if samples else [fallback]

prompt_templates = {
    "extraction": ensure_templates(samples_by_type["extraction"],
                                   "Trích xuất giá trị của trường <field> từ hóa đơn."),
    "reasoning": ensure_templates(samples_by_type["reasoning"],
                                  "Có bao nhiêu mặt hàng trên hóa đơn?"),
    "aggregation": ensure_templates(samples_by_type["aggregation"],
                                    "Tổng số tiền của hóa đơn là bao nhiêu?"),
    "stats": {
        "type_counts": dict(type_counter),
        "questions_per_image_p50": int(qpi_series.quantile(0.5)),
        "questions_per_image_p95": int(qpi_series.quantile(0.95)),
        "answer_length_p50": int(ans_series.quantile(0.5)),
        "answer_length_p95": int(ans_series.quantile(0.95)),
    },
}
(OUT / "prompt_templates.yaml").write_text(
    yaml.safe_dump(prompt_templates, sort_keys=False, allow_unicode=True), encoding="utf-8"
)
print(f"§5 wrote prompt_templates.yaml")
```

- [ ] **Step 4: Restart & Run All, verify §7 passes**

Expected: §5 prints type counts; §7 prints `§5 OK — extraction=N, reasoning=M, aggregation=K`.

- [ ] **Step 5: Commit**

```bash
git add eda.ipynb
git commit -m "feat(eda): §5 VQA structure analysis and prompt templates"
```

---

## Task 8: §6 Cross-Dataset Comparison → training_strategy.yaml

**[DEFERRED — pending VQA access]**

**Files:**
- Modify: `eda.ipynb` (§6 code cell + §7 cell extension)

- [ ] **Step 1: Extend §7 with `validate_section_6`**

Insert in §7:

```python
def validate_section_6():
    """§6 emits training_strategy.yaml with valid strategy and non-empty justification."""
    p = OUT / "training_strategy.yaml"
    assert p.exists(), f"{p} missing"
    ts = yaml.safe_load(p.read_text(encoding="utf-8"))
    assert ts.get("strategy") in ("joint", "sequential", "weighted"), \
        f"strategy must be one of joint/sequential/weighted: {ts.get('strategy')}"
    assert ts.get("justification") and len(ts["justification"]) > 10, \
        "justification must be a non-trivial string"
    if ts["strategy"] == "weighted":
        assert "dataset_weights" in ts, "weighted strategy requires dataset_weights"
    print(f"  §6 OK — strategy={ts['strategy']}")
```

Append `validate_section_6()` to `run_all_validations`.

- [ ] **Step 2: Restart & Run All, confirm §7 fails**

Expected: `AssertionError: eda_outputs/training_strategy.yaml missing`.

- [ ] **Step 3: Write §6 cell**

```python
from scipy import stats as scipy_stats

mcocr_fields = {f["name"] for f in fields if "mc_ocr" in f["source_datasets"]}
vqa_fields = {f["name"] for f in fields if "vqa" in f["source_datasets"]}
schema_jaccard = (
    len(mcocr_fields & vqa_fields) / len(mcocr_fields | vqa_fields)
    if (mcocr_fields | vqa_fields) else 0.0
)

mcocr_ls = df_img.loc[df_img["dataset"] == "mc_ocr", "longest_side"].dropna()
vqa_ls = df_img.loc[df_img["dataset"] == "vqa", "longest_side"].dropna()
ks_resolution = scipy_stats.ks_2samp(mcocr_ls, vqa_ls).pvalue if len(mcocr_ls) and len(vqa_ls) else 0.0

mcocr_tokens, vqa_tokens = [], []
for ds, ann in annotation_pairs():
    n = count_tokens(receipt_text(ann))
    (mcocr_tokens if ds == "mc_ocr" else vqa_tokens).append(n)
ks_text = scipy_stats.ks_2samp(mcocr_tokens, vqa_tokens).pvalue if mcocr_tokens and vqa_tokens else 0.0

print(f"§6 distribution overlap: schema_jaccard={schema_jaccard:.3f}, "
      f"resolution_ks_pvalue={ks_resolution:.4f}, text_ks_pvalue={ks_text:.4f}")

JACCARD_HIGH, JACCARD_LOW = 0.5, 0.2
KS_HIGH, KS_LOW = 0.05, 0.001

def decide_strategy():
    if schema_jaccard >= JACCARD_HIGH and ks_resolution >= KS_HIGH and ks_text >= KS_HIGH:
        return "joint", None, "Distributions overlap heavily across schema, resolution, and text length."
    if schema_jaccard < JACCARD_LOW and ks_resolution < KS_LOW and ks_text < KS_LOW:
        return "sequential", None, ("Distributions diverge sharply across all three measures; "
                                    "train sequentially to avoid one dataset dominating gradient updates.")
    total = MCOCR_COUNT + VQA_COUNT
    weights = {"mc_ocr": round(VQA_COUNT / total, 3), "vqa": round(MCOCR_COUNT / total, 3)}
    return "weighted", weights, (f"Mixed-overlap profile (jaccard={schema_jaccard:.2f}, "
                                 f"ks_res={ks_resolution:.3f}, ks_text={ks_text:.3f}); "
                                 f"weight inversely proportional to size to prevent the larger dataset from dominating.")

strategy, weights, justification = decide_strategy()

ts = {
    "strategy": strategy,
    "justification": justification,
    "metrics": {
        "schema_jaccard": round(schema_jaccard, 3),
        "resolution_ks_pvalue": round(ks_resolution, 4),
        "text_ks_pvalue": round(ks_text, 4),
    },
    "thresholds": {
        "jaccard_high": JACCARD_HIGH, "jaccard_low": JACCARD_LOW,
        "ks_high": KS_HIGH, "ks_low": KS_LOW,
    },
}
if weights is not None:
    ts["dataset_weights"] = weights

(OUT / "training_strategy.yaml").write_text(
    yaml.safe_dump(ts, sort_keys=False, allow_unicode=True), encoding="utf-8"
)
print(f"§6 wrote training_strategy.yaml: strategy={strategy}")
```

- [ ] **Step 4: Restart & Run All, verify §7 passes**

Expected: §6 prints overlap metrics; §7 prints `§6 OK — strategy=joint|sequential|weighted`.

- [ ] **Step 5: Commit**

```bash
git add eda.ipynb
git commit -m "feat(eda): §6 cross-dataset comparison and training strategy"
```

---

## Task 9: §7 Final Summary + Integration Verification

**Files:**
- Modify: `eda.ipynb` (§7 cell — extend with summary printing)

- [ ] **Step 1: Extend the §7 cell with a final summary printer**

Add this function ABOVE `def run_all_validations()`:

```python
def print_summary():
    """Final dump of every artifact path and a few key values for human eyeballing."""
    print("\n=== EDA Artifact Summary ===")
    for p in sorted(OUT.iterdir()):
        size = p.stat().st_size
        print(f"  {p.relative_to(OUT.parent)}  ({size} bytes)")

    caps = yaml.safe_load((OUT / "training_caps.yaml").read_text(encoding="utf-8"))
    norm = yaml.safe_load((OUT / "normalization_rules.yaml").read_text(encoding="utf-8"))
    aug  = yaml.safe_load((OUT / "augmentation.yaml").read_text(encoding="utf-8"))
    ts   = yaml.safe_load((OUT / "training_strategy.yaml").read_text(encoding="utf-8"))
    schema = json.loads((OUT / "schema.json").read_text(encoding="utf-8"))
    pt   = yaml.safe_load((OUT / "prompt_templates.yaml").read_text(encoding="utf-8"))

    print(f"\nseed: {SEED}")
    print(f"datasets: MC-OCR={MCOCR_COUNT}, VQA={VQA_COUNT}")
    print(f"mc_ocr_tree_hash: {MCOCR_TREE_HASH[:16]}...")
    print(f"\nschema fields: {len(schema['fields'])}")
    print(f"max_resolution: {caps['max_resolution']}, target_resolution: {caps['target_resolution']}")
    print(f"max_seq_length: {caps['max_seq_length']}")
    print(f"diacritic_density: {norm['diacritic_density']}")
    print(f"rotation_range_degrees: {aug['rotation_range_degrees']}")
    print(f"brightness_range: {aug['brightness_range']}")
    print(f"prompt_template counts: {pt['stats']['type_counts']}")
    print(f"training_strategy: {ts['strategy']}")
```

Modify `run_all_validations` so it ends with `print_summary()`:

```python
def run_all_validations():
    print("Running sanity checks...")
    validate_section_0()
    validate_section_1()
    validate_section_2()
    validate_section_3()
    validate_section_4()
    validate_section_5()
    validate_section_6()
    print("All sanity checks passed.")
    print_summary()

run_all_validations()
```

- [ ] **Step 2: Restart & Run All, verify the full pipeline**

`Kernel → Restart & Run All`. Expected output ends with the summary block listing all 6 artifact filenames, their sizes, and key values. Every `validate_section_N` prints its OK line.

- [ ] **Step 3: Inspect each artifact manually**

```bash
ls -la eda_outputs/
cat eda_outputs/training_caps.yaml
cat eda_outputs/training_strategy.yaml
head -40 eda_outputs/schema.json
head -30 eda_outputs/normalization_rules.yaml
head -30 eda_outputs/augmentation.yaml
head -30 eda_outputs/prompt_templates.yaml
```

Expected: every file is non-empty, values are sensible (resolution in low-thousands, seq_length in hundreds-to-low-thousands, ranges have low < high, strategy is one of three valid values).

- [ ] **Step 4: Final commit**

```bash
git add eda.ipynb
git commit -m "feat(eda): §7 final summary printer and full integration"
```

---

## Verification Checklist

After Task 9, verify:
- [ ] `eda.ipynb` runs end-to-end on a fresh kernel (`Restart & Run All`) without errors.
- [ ] `eda_outputs/` contains exactly 6 files: `schema.json`, `training_caps.yaml`, `normalization_rules.yaml`, `augmentation.yaml`, `prompt_templates.yaml`, `training_strategy.yaml`.
- [ ] §7 prints `All sanity checks passed.` followed by the artifact summary.
- [ ] Every artifact's values are within sane ranges (the §7 assertions enforce this; manual eyeballing in Task 9 Step 3 catches anything else).
- [ ] No section produced placeholders, NaN values, or empty lists in its artifact.

## Out of Scope (per spec)

- **No tests directory.** §7 is the entire regression suite.
- **No Pydantic loader.** Artifacts are plain YAML/JSON; the consumption layer is a Phase 2 concern.
- **No manual labeling.** All analyses are automated.
- **No splits artifact.** Receipt-type categorization (2.5) and annotation audit (2.6) are dropped from EDA per the brainstorm decision; the spec records the consequence (no per-receipt-type eval breakdown unless 2.5 is reintroduced).
- **No model code, training, or eval pipeline.**

If any of those need to come back, that is a *new* spec + plan, not an extension of this one.
