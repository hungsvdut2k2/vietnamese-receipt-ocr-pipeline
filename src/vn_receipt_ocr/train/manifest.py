from __future__ import annotations

import importlib.metadata
import platform
import subprocess


_TRACKED_LIBS = [
    "torch",
    "transformers",
    "peft",
    "trl",
    "accelerate",
    "bitsandbytes",
    "unsloth",
    "huggingface_hub",
    "wandb",
    "jiwer",
    "pydantic",
]


def _git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
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
    *,
    config_dict: dict,
    run_id: str,
    seed: int,
    gpu_device_name: str,
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
