from __future__ import annotations
from pathlib import Path
import re
from typing import Any
from torch.utils.data import Dataset


_SUFFIX_RE = re.compile(r"^(?P<prefix>.+)_(?P<idx>\d+)\.jpg$")


def group_lines_by_prefix(text_path: Path | str) -> dict[str, list[str]]:
    """
    Read a line-OCR text file (one row = '<filename>_<N>.jpg\\t<text>'),
    group rows by stripped-suffix prefix, and sort within each group by integer N.
    Returns {prefix: [text_for_N=0, text_for_N=1, ...]}.
    """
    text_path = Path(text_path)
    by_prefix: dict[str, list[tuple[int, str]]] = {}
    with open(text_path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n").rstrip("\r")
            if not line:
                continue
            try:
                fname, text = line.split("\t", 1)
            except ValueError as e:
                raise ValueError(f"Malformed row in {text_path}: {line!r}") from e
            m = _SUFFIX_RE.match(fname)
            if not m:
                raise ValueError(f"Filename does not match <prefix>_<N>.jpg: {fname!r}")
            prefix = m.group("prefix")
            idx = int(m.group("idx"))
            by_prefix.setdefault(prefix, []).append((idx, text))
    return {p: [t for _, t in sorted(rows, key=lambda r: r[0])]
            for p, rows in by_prefix.items()}


class MCOCRDataset(Dataset):
    """
    Each item: {image_path: Path, full_text: str, instruction: str, prefix: str}.
    Image loading is deferred to the collator/processor so we don't repeatedly
    decode the same image on retries; tests can run without images by passing
    require_images=False.
    """

    def __init__(
        self,
        text_path: Path | str,
        images_dir: Path | str,
        instruction: str,
        *,
        require_images: bool = True,
    ) -> None:
        self.text_path = Path(text_path)
        self.images_dir = Path(images_dir)
        self.instruction = instruction
        self.require_images = require_images

        self._lines_by_prefix = group_lines_by_prefix(self.text_path)
        self._prefixes = sorted(self._lines_by_prefix.keys())

    def __len__(self) -> int:
        return len(self._prefixes)

    def full_text(self, prefix: str) -> str:
        return "\n".join(self._lines_by_prefix[prefix])

    def __getitem__(self, idx: int) -> dict[str, Any]:
        prefix = self._prefixes[idx]
        image_path = self.images_dir / f"{prefix}.jpg"
        if self.require_images and not image_path.is_file():
            raise FileNotFoundError(f"Image not found for prefix '{prefix}': {image_path}")
        return {
            "prefix": prefix,
            "image_path": image_path,
            "instruction": self.instruction,
            "full_text": self.full_text(prefix),
        }
