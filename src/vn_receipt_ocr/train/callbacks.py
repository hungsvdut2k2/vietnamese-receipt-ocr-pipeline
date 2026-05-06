from __future__ import annotations

import json
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
