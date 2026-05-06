from vn_receipt_ocr.train.trainer import generate_run_id, build_run_name


def test_generate_run_id_is_stable_for_same_inputs():
    a = generate_run_id(seed=42, model_id="x", date_str="20260506")
    b = generate_run_id(seed=42, model_id="x", date_str="20260506")
    assert a == b


def test_build_run_name_uses_template():
    name = build_run_name(
        template="{model_short}-r{lora_rank}-{date}-{nnn}",
        model_id="Qwen/Qwen3-VL-2B-Instruct",
        lora_rank=16,
        date_str="20260506",
        nnn="001",
    )
    assert name == "qwen3vl2b-r16-20260506-001"
