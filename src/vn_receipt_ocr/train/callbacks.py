from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any, Callable

from vn_receipt_ocr.kaggle.secrets import get_secret_or_none


class JSONLFallback:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = open(self.path, "a", encoding="utf-8")

    def log(self, record: dict[str, Any]) -> None:
        self._fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        try:
            self._fh.close()
        except Exception:
            pass

    def __enter__(self) -> "JSONLFallback":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


def init_wandb(
    *,
    project: str,
    run_name: str,
    config: dict,
    fallback_jsonl: Path,
    mode_fallback: str = "offline",
) -> dict[str, Any]:
    """Initialize wandb if a key is present; else fall back to JSONL.

    Returns {'run': <wandb.Run|None>, 'mode': str, 'fallback': JSONLFallback|None}.
    """
    key = get_secret_or_none("WANDB_API_KEY")
    if key is None:
        if mode_fallback == "disabled":
            return {
                "run": None,
                "mode": "disabled",
                "fallback": JSONLFallback(path=fallback_jsonl),
            }
        os.environ["WANDB_MODE"] = "offline"
    else:
        os.environ["WANDB_API_KEY"] = key

    try:
        import wandb  # type: ignore
        run = wandb.init(project=project, name=run_name, config=config)
        return {
            "run": run,
            "mode": os.environ.get("WANDB_MODE", "online"),
            "fallback": None,
        }
    except Exception:
        return {
            "run": None,
            "mode": "jsonl",
            "fallback": JSONLFallback(path=fallback_jsonl),
        }


def log_event(state: dict, payload: dict) -> None:
    """Single entry point for emitting metrics to wandb or jsonl fallback."""
    run = state.get("run")
    fb: JSONLFallback | None = state.get("fallback")
    if run is not None:
        run.log(payload)
    if fb is not None:
        fb.log(payload)


class PerEpochEvalCallback:
    """A minimal trainer-callback shape (not subclassed from
    transformers.TrainerCallback so it can be unit-tested without HF imports).
    The training wrapper adapts it via a thin shim."""

    def __init__(
        self, *, eval_fn: Callable[[], dict], log_event_fn: Callable[[dict], None]
    ) -> None:
        self.eval_fn = eval_fn
        self.log_event_fn = log_event_fn
        self.history: list[dict] = []

    def on_epoch_end(self, args, state, control, **kwargs):
        metrics = self.eval_fn()
        epoch = getattr(state, "epoch", None)
        record = {"epoch": epoch, **{f"eval/{k}": v for k, v in metrics.items()}}
        self.history.append(record)
        self.log_event_fn(record)


class CheckpointSyncCallback:
    """On each eval, optionally write/save the model and upload to HF Hub
    only if `primary_metric` improved.

    save_fn(path): persists model+processor to `path`.
    upload_fn(local_dir, commit_msg): pushes to HF Hub (mocked in tests).
    """

    def __init__(
        self,
        *,
        local_root: Path,
        save_fn: Callable[[Path], None],
        upload_fn: Callable[[Path, str], None] | None,
        primary_metric: str = "diacritic_cer",
    ) -> None:
        self.local_root = Path(local_root)
        self.local_root.mkdir(parents=True, exist_ok=True)
        self.save_fn = save_fn
        self.upload_fn = upload_fn
        self.primary_metric = primary_metric
        self.best_value = math.inf

    def handle_eval(self, *, epoch: int | float | None, metrics: dict) -> bool:
        epoch_dir = self.local_root / f"epoch_{int(epoch) if epoch else 0}"
        self.save_fn(epoch_dir)

        cur = metrics.get(self.primary_metric)
        if cur is None:
            return False

        if cur < self.best_value:
            self.best_value = cur
            best_dir = self.local_root / "best"
            self.save_fn(best_dir)
            if self.upload_fn is not None:
                self.upload_fn(
                    best_dir,
                    f"epoch {epoch} | {self.primary_metric} {cur:.4f}",
                )
            return True
        return False
