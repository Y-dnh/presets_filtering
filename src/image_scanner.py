from __future__ import annotations

from pathlib import Path
from typing import List

from tqdm import tqdm


def scan_images(input_dir: Path, extensions: List[str]) -> List[Path]:
    """Recursively find all image files under *input_dir*.

    Returns a sorted list of absolute paths whose suffix (case-insensitive)
    matches one of *extensions*.
    """
    ext_set = {e.lower() for e in extensions}
    images: List[Path] = []

    iterator = input_dir.rglob("*")
    for path in tqdm(iterator, desc="  Сканування файлів", unit="item"):
        if path.is_file() and path.suffix.lower() in ext_set:
            images.append(path.resolve())

    images.sort()
    return images
