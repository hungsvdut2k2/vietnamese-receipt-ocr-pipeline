import pytest

from vn_receipt_ocr.config.validation import (
    validate_dtype_hardware, project_wallclock_hours, ConfigValidationError,
)


def test_bf16_on_p100_rejected():
    with pytest.raises(ConfigValidationError, match="P100"):
        validate_dtype_hardware(gpu_name="p100_16gb", dtype="bf16")


def test_fp16_on_p100_ok():
    validate_dtype_hardware(gpu_name="p100_16gb", dtype="fp16")  # no raise


def test_bf16_on_t4_ok():
    validate_dtype_hardware(gpu_name="t4_16gb", dtype="bf16")


def test_wallclock_projection_within_budget():
    h = project_wallclock_hours(
        n_train_samples=922, epochs=3, per_device_batch_size=1, grad_accum=8,
        seconds_per_optimizer_step=4.0,
    )
    # 922/8 = 115.25 steps/epoch * 3 = 345.75 steps * 4s = 1383s = 0.384h
    assert 0.3 < h < 0.5


def test_wallclock_projection_exceeds_budget_raises():
    with pytest.raises(ConfigValidationError, match="wall-clock"):
        project_wallclock_hours(
            n_train_samples=922, epochs=100,
            per_device_batch_size=1, grad_accum=8,
            seconds_per_optimizer_step=4.0,
            budget_hours=8.0, raise_on_exceed=True,
        )
