from __future__ import annotations
import hashlib
import os
import random
from datetime import datetime
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from vn_receipt_ocr.config.models import TrainConfig
from vn_receipt_ocr.config.validation import (
    validate_dtype_hardware, project_wallclock_hours,
)
from vn_receipt_ocr.data.dataset import MCOCRDataset
from vn_receipt_ocr.data.collator import QwenVLCollator
from vn_receipt_ocr.eval.aggregate import aggregate_metrics
from vn_receipt_ocr.eval.batch_predict import batch_predict
from vn_receipt_ocr.kaggle.paths import working_dir
from vn_receipt_ocr.kaggle.secrets import get_secret_or_none
from vn_receipt_ocr.model.loader import load_model_and_processor
from vn_receipt_ocr.train.callbacks import (
    init_wandb, log_event, PerEpochEvalCallback, CheckpointSyncCallback,
)
from vn_receipt_ocr.train.manifest import build_manifest


def _short_model_id(model_id: str) -> str:
    base = model_id.split("/")[-1].lower()
    base = base.replace("-instruct", "")
    return "".join(c for c in base if c.isalnum())  # qwen3-vl-2b → qwen3vl2b


def build_run_name(
    *, template: str, model_id: str, lora_rank: int, date_str: str, nnn: str
) -> str:
    return template.format(
        model_short=_short_model_id(model_id),
        lora_rank=lora_rank,
        date=date_str,
        nnn=nnn,
    )


def generate_run_id(*, seed: int, model_id: str, date_str: str) -> str:
    h = hashlib.sha1(f"{seed}-{model_id}-{date_str}".encode()).hexdigest()[:8]
    return h


def set_global_seed(seed: int, *, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")


def train(config: TrainConfig, *, dry_run: bool = False, max_steps: int | None = None) -> dict:
    """Top-level training entry point.

    dry_run=True + max_steps=1 is the smoke-test mode.
    """
    validate_dtype_hardware(
        gpu_name=config.gpu_profile.name, dtype=config.gpu_profile.dtype,
    )

    set_global_seed(config.trainer.seed, deterministic=config.trainer.deterministic)
    date_str = datetime.utcnow().strftime("%Y%m%d")
    run_id = generate_run_id(
        seed=config.trainer.seed, model_id=config.model.model_id, date_str=date_str,
    )
    run_name = build_run_name(
        template=config.wandb.run_name_template,
        model_id=config.model.model_id,
        lora_rank=config.lora.rank,
        date_str=date_str,
        nnn="001",
    )

    train_ds = MCOCRDataset(
        text_path=config.data.train_path,
        images_dir=config.data.train_images_dir,
        instruction=config.data.instruction,
    )
    val_ds = MCOCRDataset(
        text_path=config.data.val_path,
        images_dir=config.data.val_images_dir,
        instruction=config.data.instruction,
    )

    project_wallclock_hours(
        n_train_samples=len(train_ds),
        epochs=config.trainer.epochs,
        per_device_batch_size=config.trainer.per_device_batch_size,
        grad_accum=config.trainer.grad_accum,
        seconds_per_optimizer_step=4.0,
        budget_hours=config.trainer.wallclock_budget_hours,
        raise_on_exceed=not dry_run,
    )

    work = working_dir()
    wandb_state = init_wandb(
        project=config.wandb.project, run_name=run_name,
        config=config.model_dump(),
        fallback_jsonl=work / "wandb_offline.jsonl",
        mode_fallback=config.wandb.mode_fallback,
    )
    manifest = build_manifest(
        config_dict=config.model_dump(), run_id=run_id, seed=config.trainer.seed,
        gpu_device_name=(torch.cuda.get_device_name(0) if torch.cuda.is_available()
                         else "cpu"),
    )
    log_event(wandb_state, {"manifest": manifest})

    model, processor = load_model_and_processor(
        model=config.model, gpu_profile=config.gpu_profile, lora=config.lora,
    )
    collator = QwenVLCollator(processor=processor)

    from trl import SFTConfig, SFTTrainer  # type: ignore

    sft_args = SFTConfig(
        output_dir=str(work / "checkpoints"),
        per_device_train_batch_size=config.trainer.per_device_batch_size,
        gradient_accumulation_steps=config.trainer.grad_accum,
        num_train_epochs=config.trainer.epochs,
        learning_rate=config.trainer.lr,
        warmup_ratio=config.trainer.warmup_ratio,
        optim=config.trainer.optimizer,
        gradient_checkpointing=config.trainer.gradient_checkpointing,
        bf16=(config.gpu_profile.dtype == "bf16"),
        fp16=(config.gpu_profile.dtype == "fp16"),
        seed=config.trainer.seed,
        max_steps=(max_steps if max_steps is not None else -1),
        logging_steps=10,
        report_to=[],
        save_strategy="no",
    )

    def _eval_now() -> dict:
        items = [val_ds[i] for i in range(len(val_ds))]
        preds, latencies = batch_predict(
            model=model, processor=processor, items=items,
            instruction=config.data.instruction,
            max_new_tokens=config.eval.max_new_tokens,
        )
        refs = [it["full_text"] for it in items]
        return aggregate_metrics(
            predictions=preds, references=refs,
            latency_ms=latencies, metrics_enabled=config.eval.metrics_enabled,
        )

    def _save_to(path: Path) -> None:
        path.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(path)
        processor.save_pretrained(path)

    upload_fn: Callable[[Path, str], None] | None = None
    if config.hf_hub.enabled and get_secret_or_none("HF_TOKEN"):
        from huggingface_hub import HfApi  # lazy
        api = HfApi(token=get_secret_or_none("HF_TOKEN"))
        repo_name = config.hf_hub.repo_name_template.format(run_id=run_id)
        repo_id = f"{config.hf_hub.repo_owner}/{repo_name}"
        api.create_repo(repo_id=repo_id, private=config.hf_hub.private, exist_ok=True)

        def upload_fn(local_dir: Path, msg: str) -> None:
            api.upload_folder(folder_path=str(local_dir), repo_id=repo_id,
                              commit_message=msg)

    eval_cb = PerEpochEvalCallback(
        eval_fn=_eval_now,
        log_event_fn=lambda payload: log_event(wandb_state, payload),
    )
    ckpt_cb = CheckpointSyncCallback(
        local_root=work / "checkpoints",
        save_fn=_save_to, upload_fn=upload_fn,
        primary_metric="diacritic_cer",
    )

    from transformers import TrainerCallback

    class _HFEvalShim(TrainerCallback):
        def on_epoch_end(self, args, state, control, **kw):
            metrics = eval_cb.on_epoch_end(args, state, control, **kw) or {}
            if eval_cb.history:
                last = eval_cb.history[-1]
                clean = {k.replace("eval/", ""): v for k, v in last.items()
                         if k.startswith("eval/")}
                ckpt_cb.handle_eval(epoch=state.epoch, metrics=clean)

    trainer = SFTTrainer(
        model=model,
        args=sft_args,
        train_dataset=train_ds,
        data_collator=collator,
        callbacks=[_HFEvalShim()],
    )

    if not dry_run:
        trainer.train()

    return {
        "run_id": run_id,
        "run_name": run_name,
        "best_diacritic_cer": ckpt_cb.best_value,
        "history": eval_cb.history,
    }
