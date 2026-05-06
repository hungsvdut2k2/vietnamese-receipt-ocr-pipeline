from __future__ import annotations


_NAME_TO_PROFILE: list[tuple[str, str]] = [
    ("p100", "p100_16gb"),
    ("t4", "t4_16gb"),
    ("l4", "l4_24gb"),
    ("a100", "a100_40gb"),
    ("v100", "v100_16gb"),
]


def detect_gpu_profile_name(
    *, device_name: str | None = None, default: str = "p100_16gb"
) -> str:
    if device_name is None:
        try:
            import torch

            device_name = (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else ""
            )
        except Exception:
            device_name = ""
    name = (device_name or "").lower()
    for needle, profile in _NAME_TO_PROFILE:
        if needle in name:
            return profile
    return default
