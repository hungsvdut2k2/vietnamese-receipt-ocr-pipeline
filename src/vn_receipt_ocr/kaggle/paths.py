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
