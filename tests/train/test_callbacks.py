from __future__ import annotations

import json
import sys


def test_init_wandb_returns_offline_when_no_key(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "vn_receipt_ocr.train.callbacks.get_secret_or_none",
        lambda key: None,
    )
    # Make wandb import fail to avoid actual wandb.init side effects
    monkeypatch.setitem(sys.modules, "wandb", None)
    monkeypatch.delenv("WANDB_MODE", raising=False)

    from vn_receipt_ocr.train.callbacks import init_wandb

    out = init_wandb(
        project="test-proj",
        run_name="r1",
        config={"a": 1},
        fallback_jsonl=tmp_path / "fb.jsonl",
        mode_fallback="offline",
    )
    # Either WANDB_MODE is offline OR we use JSONL fallback
    assert out["mode"] in ("offline", "disabled", "jsonl")
    # When the key is absent, the fallback object must be created and usable.
    assert out["fallback"] is not None
    out["fallback"].log({"step": 0})
    out["fallback"].close()


def test_jsonl_fallback_writes_lines(tmp_path):
    from vn_receipt_ocr.train.callbacks import JSONLFallback

    f = tmp_path / "fb.jsonl"
    fb = JSONLFallback(path=f)
    fb.log({"loss": 1.0, "step": 1})
    fb.log({"loss": 0.5, "step": 2})
    lines = f.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 2
    assert json.loads(lines[0])["loss"] == 1.0


def test_per_epoch_eval_callback_calls_eval_fn(tmp_path):
    from vn_receipt_ocr.train.callbacks import PerEpochEvalCallback

    eval_calls = []
    logged: list[dict] = []

    def fake_eval():
        eval_calls.append(1)
        return {"cer": 0.1}

    cb = PerEpochEvalCallback(eval_fn=fake_eval, log_event_fn=logged.append)
    # Simulate HF Trainer "on_epoch_end" hook.
    cb.on_epoch_end(args=None, state=type("S", (), {"epoch": 1})(), control=None)
    assert eval_calls == [1]
    assert len(logged) == 1
    record = logged[0]
    assert record["epoch"] == 1
    assert record["eval/cer"] == 0.1
    assert cb.history == [record]


def test_checkpoint_sync_uploads_only_when_better(tmp_path):
    from vn_receipt_ocr.train.callbacks import CheckpointSyncCallback
    uploads = []

    def fake_upload(local_dir, commit_msg):
        uploads.append((local_dir, commit_msg))

    def fake_save(path):
        path.mkdir(parents=True, exist_ok=True)

    cb = CheckpointSyncCallback(
        local_root=tmp_path / "ckpt",
        save_fn=fake_save,
        upload_fn=fake_upload,
        primary_metric="diacritic_cer",
    )
    # First call: 0.5 → improves over inf → upload
    cb.handle_eval(epoch=1, metrics={"diacritic_cer": 0.5})
    assert len(uploads) == 1
    # Second call: 0.6 → worse → no upload
    cb.handle_eval(epoch=2, metrics={"diacritic_cer": 0.6})
    assert len(uploads) == 1
    # Third call: 0.4 → improves → upload
    cb.handle_eval(epoch=3, metrics={"diacritic_cer": 0.4})
    assert len(uploads) == 2
    # Fourth call: equal to current best → strict improvement only, no upload.
    cb.handle_eval(epoch=4, metrics={"diacritic_cer": 0.4})
    assert len(uploads) == 2
    # Fifth call: missing primary metric → returns False, no upload.
    cb.handle_eval(epoch=5, metrics={"other": 0.0})
    assert len(uploads) == 2


def test_checkpoint_sync_handles_no_upload_fn(tmp_path):
    from vn_receipt_ocr.train.callbacks import CheckpointSyncCallback

    saved: list = []

    def fake_save(path):
        path.mkdir(parents=True, exist_ok=True)
        saved.append(path)

    cb = CheckpointSyncCallback(
        local_root=tmp_path / "ckpt",
        save_fn=fake_save,
        upload_fn=None,
        primary_metric="diacritic_cer",
    )
    improved = cb.handle_eval(epoch=1, metrics={"diacritic_cer": 0.5})
    assert improved is True
    # Both epoch_dir and best/ written; no upload attempted.
    assert any(p.name == "best" for p in saved)
