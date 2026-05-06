from __future__ import annotations
from pathlib import Path
import json
from typing import Any

from vn_receipt_ocr.config.models import TrainConfig
from vn_receipt_ocr.config.loader import load_train_config


def train(
    config: TrainConfig | None = None,
    *,
    config_path: Path | str | None = None,
    overrides: dict[str, Any] | None = None,
    dry_run: bool = False,
    max_steps: int | None = None,
) -> dict:
    from vn_receipt_ocr.train.trainer import train as _train
    if config is None:
        if config_path is None:
            raise ValueError("provide either config or config_path")
        config = load_train_config(
            experiment_path=config_path, configs_root="configs",
            overrides=overrides or {},
        )
    return _train(config, dry_run=dry_run, max_steps=max_steps)


def evaluate(
    config: TrainConfig, *, adapters: str, split: str = "val"
) -> dict:
    from vn_receipt_ocr.data.dataset import MCOCRDataset
    from vn_receipt_ocr.model.loader import load_model_and_processor
    from vn_receipt_ocr.eval.batch_predict import batch_predict
    from vn_receipt_ocr.eval.aggregate import aggregate_metrics

    text_path = config.data.val_path if split == "val" else config.data.test_path
    images_dir = config.data.val_images_dir if split == "val" else config.data.test_images_dir
    if not text_path:
        raise ValueError(f"No {split} path configured.")

    ds = MCOCRDataset(text_path=text_path, images_dir=images_dir,
                      instruction=config.data.instruction)
    items = [ds[i] for i in range(len(ds))]

    model, processor = load_model_and_processor(
        model=config.model, gpu_profile=config.gpu_profile, lora=config.lora,
    )
    # Load adapters into the freshly-loaded model
    if adapters.startswith("hf://"):
        from huggingface_hub import snapshot_download
        local = snapshot_download(repo_id=adapters[len("hf://"):])
    else:
        local = adapters
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, local)

    preds, lat = batch_predict(
        model=model, processor=processor, items=items,
        instruction=config.data.instruction,
        max_new_tokens=config.eval.max_new_tokens,
    )
    refs = [it["full_text"] for it in items]
    return aggregate_metrics(
        predictions=preds, references=refs, latency_ms=lat,
        metrics_enabled=config.eval.metrics_enabled,
    )


def predict(
    config: TrainConfig, *, adapters: str, inputs: Path | str, output: Path | str,
) -> None:
    from vn_receipt_ocr.model.loader import load_model_and_processor
    from vn_receipt_ocr.eval.batch_predict import batch_predict

    inputs = Path(inputs)
    if inputs.is_dir():
        files = sorted(p for p in inputs.iterdir() if p.suffix.lower() == ".jpg")
    else:
        files = [inputs]
    items = [{"image_path": p, "instruction": config.data.instruction,
              "full_text": ""} for p in files]

    model, processor = load_model_and_processor(
        model=config.model, gpu_profile=config.gpu_profile, lora=config.lora,
    )
    if adapters.startswith("hf://"):
        from huggingface_hub import snapshot_download
        local = snapshot_download(repo_id=adapters[len("hf://"):])
    else:
        local = adapters
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, local)

    preds, _ = batch_predict(
        model=model, processor=processor, items=items,
        instruction=config.data.instruction,
        max_new_tokens=config.eval.max_new_tokens,
    )
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as f:
        for p, t in zip(files, preds):
            f.write(json.dumps({"image": str(p), "prediction": t},
                              ensure_ascii=False) + "\n")


__all__ = ["train", "evaluate", "predict", "TrainConfig", "load_train_config"]
