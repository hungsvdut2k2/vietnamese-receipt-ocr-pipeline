from __future__ import annotations


class ConfigValidationError(ValueError):
    pass


_BF16_INCOMPATIBLE_GPUS = {"p100", "k80", "v100"}


def validate_dtype_hardware(gpu_name: str, dtype: str) -> None:
    """Raise ConfigValidationError if dtype is not supported on the given GPU."""
    name = gpu_name.lower()
    if dtype == "bf16" and any(g in name for g in _BF16_INCOMPATIBLE_GPUS):
        raise ConfigValidationError(
            f"GPU profile '{gpu_name}' does not support BF16 (P100/V100/K80 lack "
            "BF16 support); use FP16 instead."
        )


def project_wallclock_hours(
    *,
    n_train_samples: int,
    epochs: int,
    per_device_batch_size: int,
    grad_accum: int,
    seconds_per_optimizer_step: float,
    budget_hours: float | None = None,
    raise_on_exceed: bool = False,
) -> float:
    """Compute projected training wall-clock; optionally raise if it exceeds budget."""
    effective_batch = per_device_batch_size * grad_accum
    steps_per_epoch = (n_train_samples + effective_batch - 1) // effective_batch
    total_steps = steps_per_epoch * epochs
    seconds = total_steps * seconds_per_optimizer_step
    hours = seconds / 3600.0
    if raise_on_exceed and budget_hours is not None and hours > budget_hours:
        raise ConfigValidationError(
            f"Projected wall-clock {hours:.2f}h exceeds budget {budget_hours:.2f}h. "
            f"Reduce epochs or seconds_per_optimizer_step."
        )
    return hours
