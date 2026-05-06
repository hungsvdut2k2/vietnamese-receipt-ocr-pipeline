from __future__ import annotations
import argparse
import json
from pathlib import Path

from vn_receipt_ocr.config.loader import load_train_config


def _parse_overrides(items: list[str]) -> dict[str, object]:
    out: dict[str, object] = {}
    for it in items:
        if "=" not in it:
            raise SystemExit(f"--override expects key=value, got {it!r}")
        key, _, val = it.partition("=")
        coerced: object
        if val.lower() in ("true", "false"):
            coerced = val.lower() == "true"
        else:
            try:
                coerced = int(val)
            except ValueError:
                try:
                    coerced = float(val)
                except ValueError:
                    coerced = val
        out[key] = coerced
    return out


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vn_receipt_ocr",
                                description="Vietnamese Receipt OCR (Qwen3-VL-2B + Unsloth + LoRA)")
    sub = p.add_subparsers(dest="command", required=True)

    pt = sub.add_parser("train", help="Train a LoRA adapter")
    pt.add_argument("--config", required=True, type=Path)
    pt.add_argument("--configs-root", default="configs", type=Path)
    pt.add_argument("--override", action="append", default=[],
                    help="Dotted-key overrides, e.g. lora.rank=8")
    pt.add_argument("--dry-run", action="store_true")
    pt.add_argument("--max-steps", type=int, default=None)

    pe = sub.add_parser("eval", help="Run evaluation on val/test")
    pe.add_argument("--config", required=True, type=Path)
    pe.add_argument("--configs-root", default="configs", type=Path)
    pe.add_argument("--adapters", required=True, type=str,
                    help="Local path or hf://owner/repo")
    pe.add_argument("--override", action="append", default=[])
    pe.add_argument("--split", choices=["val", "test"], default="val")

    pp = sub.add_parser("predict", help="Predict on a directory of images")
    pp.add_argument("--config", required=True, type=Path)
    pp.add_argument("--configs-root", default="configs", type=Path)
    pp.add_argument("--adapters", required=True, type=str)
    pp.add_argument("--inputs", required=True, type=Path)
    pp.add_argument("--output", required=True, type=Path)
    pp.add_argument("--override", action="append", default=[])
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    overrides = _parse_overrides(args.override)
    cfg = load_train_config(
        experiment_path=args.config, configs_root=args.configs_root,
        overrides=overrides,
    )

    if args.command == "train":
        from vn_receipt_ocr.train.trainer import train as _train
        result = _train(cfg, dry_run=args.dry_run, max_steps=args.max_steps)
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.command == "eval":
        from vn_receipt_ocr import evaluate
        result = evaluate(cfg, adapters=args.adapters, split=args.split)
        print(json.dumps(result, indent=2, default=str))
        return 0

    if args.command == "predict":
        from vn_receipt_ocr import predict
        predict(cfg, adapters=args.adapters, inputs=args.inputs,
                output=args.output)
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
