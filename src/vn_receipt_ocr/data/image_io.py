from __future__ import annotations

from pathlib import Path

from PIL import Image


def load_image_rgb(image_path: Image.Image | str | Path) -> Image.Image:
    if isinstance(image_path, Image.Image):
        return image_path.convert("RGB")
    return Image.open(image_path).convert("RGB")
