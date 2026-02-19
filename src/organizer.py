from __future__ import annotations

import shutil
from pathlib import Path
from typing import List

import numpy as np
from tqdm import tqdm

from .config import AnnotationsConfig


def _find_label_path(
    image_path: Path,
    ann_cfg: AnnotationsConfig,
) -> Path | None:
    """Find the corresponding YOLO label file for an image.

    Replaces the image_subdir part of the path with label_subdir
    and swaps the extension.
    """
    parts = image_path.parts
    img_sub = ann_cfg.image_subdir

    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == img_sub:
            label_parts = parts[:i] + (ann_cfg.label_subdir,) + parts[i + 1:]
            label_path = Path(*label_parts).with_suffix(ann_cfg.label_extension)
            if label_path.exists():
                return label_path
            return None

    return None


def _transfer_file(src: Path, dest: Path, mode: str) -> None:
    if dest.exists():
        stem = dest.stem
        suffix = dest.suffix
        parent = dest.parent
        counter = 1
        while dest.exists():
            dest = parent / f"{stem}_{counter}{suffix}"
            counter += 1

    if mode == "move":
        shutil.move(str(src), str(dest))
    else:
        shutil.copy2(str(src), str(dest))


def organize_files(
    image_paths: List[Path],
    labels: np.ndarray,
    output_dir: Path,
    mode: str,
    ann_cfg: AnnotationsConfig | None = None,
) -> None:
    """Copy or move images (and annotations) into preset folders.

    When annotations are enabled, output structure per preset:
        preset_NNN/img/image.jpg
        preset_NNN/lab/image.txt

    When annotations are disabled:
        preset_NNN/image.jpg
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    use_annotations = ann_cfg is not None and ann_cfg.enabled
    n_labels_found = 0
    n_labels_missing = 0

    for path, label in tqdm(
        zip(image_paths, labels), total=len(image_paths), desc="  Організація файлів", unit="file"
    ):
        folder_name = "noise" if label == -1 else f"preset_{label:03d}"

        if use_annotations:
            img_dest_dir = output_dir / folder_name / ann_cfg.image_subdir
            lab_dest_dir = output_dir / folder_name / ann_cfg.label_subdir
        else:
            img_dest_dir = output_dir / folder_name
            lab_dest_dir = None

        img_dest_dir.mkdir(parents=True, exist_ok=True)

        _transfer_file(path, img_dest_dir / path.name, mode)

        if use_annotations:
            label_path = _find_label_path(path, ann_cfg)
            if label_path is not None:
                lab_dest_dir.mkdir(parents=True, exist_ok=True)
                dest_label_name = path.stem + ann_cfg.label_extension
                _transfer_file(label_path, lab_dest_dir / dest_label_name, mode)
                n_labels_found += 1
            else:
                n_labels_missing += 1

    if use_annotations:
        print(f"  Анотацій знайдено: {n_labels_found}, не знайдено: {n_labels_missing}")
