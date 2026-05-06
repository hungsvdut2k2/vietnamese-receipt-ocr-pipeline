from pathlib import Path
import yaml  # noqa: F401

from vn_receipt_ocr.config.loader import deep_merge, load_yaml, load_train_config


def test_deep_merge_overrides_leaves():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"c": 99}, "e": 5}
    assert deep_merge(base, override) == {"a": 1, "b": {"c": 99, "d": 3}, "e": 5}


def test_deep_merge_lists_are_replaced_not_concatenated():
    base = {"x": [1, 2, 3]}
    override = {"x": [9]}
    assert deep_merge(base, override) == {"x": [9]}


def test_load_yaml_round_trip(tmp_path: Path):
    p = tmp_path / "x.yaml"
    p.write_text("a: 1\nb:\n  c: 2\n", encoding="utf-8")
    assert load_yaml(p) == {"a": 1, "b": {"c": 2}}


def test_load_train_config_composes_includes(tmp_path: Path):
    base_dir = tmp_path
    (base_dir / "gpu_profiles").mkdir()
    (base_dir / "data").mkdir()
    (base_dir / "model").mkdir()
    (base_dir / "lora").mkdir()
    (base_dir / "experiments").mkdir()

    (base_dir / "gpu_profiles" / "p100_16gb.yaml").write_text(
        "gpu_profile:\n  name: p100_16gb\n  dtype: fp16\n  vram_gb: 16\n  recommended_batch_size: 1\n"
    )
    (base_dir / "data" / "mcocr.yaml").write_text(
        "data:\n  train_path: a\n  train_images_dir: b\n  val_path: c\n  val_images_dir: d\n"
        "  instruction: 'X'\n  target_resolution: 888\n  max_resolution: 1388\n  max_seq_length: 203\n"
    )
    (base_dir / "model" / "q3.yaml").write_text(
        "model:\n  model_id: Qwen/Qwen3-VL-2B-Instruct\n"
    )
    (base_dir / "lora" / "r16.yaml").write_text(
        "lora:\n  rank: 16\n  alpha: 32\n  dropout: 0.05\n"
        "  target_modules: [q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]\n  bias: none\n"
        "  finetune_vision_layers: false\n"
    )
    (base_dir / "experiments" / "exp.yaml").write_text(
        "include:\n  - gpu_profiles/p100_16gb.yaml\n  - data/mcocr.yaml\n"
        "  - model/q3.yaml\n  - lora/r16.yaml\n"
        "trainer:\n  epochs: 3\n  per_device_batch_size: 1\n  grad_accum: 8\n  lr: 1e-4\n"
        "  warmup_ratio: 0.05\n  optimizer: adamw_8bit\n  gradient_checkpointing: true\n"
        "  seed: 42\n  deterministic: false\n  wallclock_budget_hours: 8.0\n"
        "wandb:\n  project: vn-receipt-ocr\n  run_name_template: '{model_short}-r{lora_rank}-{date}-{nnn}'\n"
        "  enabled: true\n  mode_fallback: offline\n"
        "hf_hub:\n  repo_owner: me\n  repo_name_template: 'vn-receipt-ocr-{run_id}'\n"
        "  private: true\n  enabled: true\n"
        "eval:\n  metrics_enabled: [cer, diacritic_cer, cer_normalized, wer, edit_ops, length_ratio, empty_pred_rate, latency]\n"
        "  decode: greedy\n  max_new_tokens: 244\n  sample_table_size: 10\n"
    )

    cfg = load_train_config(base_dir / "experiments" / "exp.yaml", configs_root=base_dir)
    assert cfg.lora.rank == 16
    assert cfg.gpu_profile.name == "p100_16gb"
    assert cfg.model.model_id == "Qwen/Qwen3-VL-2B-Instruct"


def test_overrides_take_highest_precedence(tmp_path: Path):
    # same setup as above, with overrides={"lora.rank": 32}
    pass
