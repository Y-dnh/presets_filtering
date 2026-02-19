from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from .config import CacheConfig, ModelConfig, PreprocessingConfig

# Avoid OpenCV thread oversubscription inside PyTorch DataLoader workers.
cv2.setNumThreads(0)


def _resolve_device(device_cfg: str) -> torch.device:
    if device_cfg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_cfg)


def _build_transform(image_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def _load_model(name: str, device: torch.device) -> torch.nn.Module:
    if name.startswith("dinov2"):
        model = torch.hub.load("facebookresearch/dinov2", name)
    elif name == "resnet50":
        from torchvision.models import resnet50, ResNet50_Weights
        base = resnet50(weights=ResNet50_Weights.DEFAULT)
        model = torch.nn.Sequential(*list(base.children())[:-1], torch.nn.Flatten())
    else:
        raise ValueError(f"Невідома модель: {name}")

    model = model.to(device)
    model.eval()
    return model


def _to_gray(img: Image.Image) -> np.ndarray:
    arr = np.array(img)
    if arr.ndim == 3:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return arr


def _gray_to_pil(gray: np.ndarray) -> Image.Image:
    return Image.fromarray(cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB))


def _trim(gray: np.ndarray, top: float, bottom: float, left: float, right: float) -> np.ndarray:
    h, w = gray.shape[:2]
    y1 = int(h * top)
    y2 = h - int(h * bottom)
    x1 = int(w * left)
    x2 = w - int(w * right)
    return gray[y1:y2, x1:x2]


def _equalize_gray(gray: np.ndarray, method: str, clip_limit: float, grid_size: int) -> np.ndarray:
    if method == "clahe":
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size, grid_size))
        return clahe.apply(gray)
    if method == "he":
        return cv2.equalizeHist(gray)
    return gray


def _preprocess_gray(gray: np.ndarray, preproc_cfg: PreprocessingConfig, inverted: bool) -> Image.Image:
    has_trim = (preproc_cfg.trim_top > 0 or preproc_cfg.trim_bottom > 0
                or preproc_cfg.trim_left > 0 or preproc_cfg.trim_right > 0)
    if has_trim:
        gray = _trim(gray, preproc_cfg.trim_top, preproc_cfg.trim_bottom,
                      preproc_cfg.trim_left, preproc_cfg.trim_right)

    if inverted:
        gray = 255 - gray

    if preproc_cfg.equalization != "none":
        gray = _equalize_gray(
            gray, preproc_cfg.equalization,
            preproc_cfg.clahe_clip_limit, preproc_cfg.clahe_grid_size,
        )

    return _gray_to_pil(gray)


class _ImageDataset(Dataset):
    """Dataset that preprocesses images in parallel via DataLoader workers.

    Each item returns either 1 tensor (normal) or 2 tensors (normal + inverted)
    depending on dual-pass mode.
    """

    def __init__(
        self,
        paths: List[Path],
        preproc_cfg: PreprocessingConfig,
        transform: transforms.Compose,
        dual: bool,
        image_size: int,
    ):
        self.paths = paths
        self.preproc_cfg = preproc_cfg
        self.transform = transform
        self.dual = dual
        self.zero = torch.zeros(3, image_size, image_size)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        try:
            img = Image.open(self.paths[idx])
            if img.mode != "RGB":
                img = img.convert("RGB")
            gray = _to_gray(img)

            normal_pil = _preprocess_gray(gray, self.preproc_cfg, inverted=False)
            t_normal = self.transform(normal_pil)

            if self.dual:
                inv_pil = _preprocess_gray(gray, self.preproc_cfg, inverted=True)
                t_inv = self.transform(inv_pil)
                return t_normal, t_inv
            return (t_normal,)
        except Exception:
            if self.dual:
                return self.zero, self.zero
            return (self.zero,)


def _compute_cache_key(
    image_paths: List[Path],
    model_cfg: ModelConfig,
    preproc_cfg: PreprocessingConfig,
) -> str:
    payload = {
        "model_name": model_cfg.name,
        "image_size": model_cfg.image_size,
        "equalization": preproc_cfg.equalization,
        "clahe_clip_limit": preproc_cfg.clahe_clip_limit,
        "clahe_grid_size": preproc_cfg.clahe_grid_size,
        "polarity_invariant": preproc_cfg.polarity_invariant,
        "l2_normalize": preproc_cfg.l2_normalize,
        "trim": [preproc_cfg.trim_top, preproc_cfg.trim_bottom,
                 preproc_cfg.trim_left, preproc_cfg.trim_right],
        "files": [str(p) for p in image_paths],
    }
    raw = json.dumps(payload, sort_keys=True).encode()
    return hashlib.sha256(raw).hexdigest()[:16]


def _try_load_cache(cache_cfg: CacheConfig, cache_key: str) -> np.ndarray | None:
    if not cache_cfg.enabled:
        return None
    cache_path = Path(cache_cfg.cache_dir) / f"features_{cache_key}.npy"
    if cache_path.exists():
        return np.load(cache_path)
    return None


def _save_cache(cache_cfg: CacheConfig, cache_key: str, features: np.ndarray) -> None:
    if not cache_cfg.enabled:
        return
    cache_dir = Path(cache_cfg.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / f"features_{cache_key}.npy", features)


def extract_features(
    image_paths: List[Path],
    model_cfg: ModelConfig,
    preproc_cfg: PreprocessingConfig,
    cache_cfg: CacheConfig,
) -> np.ndarray:
    """Extract feature vectors for all images using parallel preprocessing.

    Preprocessing (load, trim, CLAHE, transform) runs on multiple CPU workers
    via DataLoader, while GPU inference runs on the main thread. This keeps
    the GPU saturated instead of waiting for single-threaded CPU work.
    """
    cache_key = _compute_cache_key(image_paths, model_cfg, preproc_cfg)
    cached = _try_load_cache(cache_cfg, cache_key)
    if cached is not None:
        print(f"  Завантажено з кешу ({cached.shape[0]} зображень, {cached.shape[1]}-dim)")
        return cached

    device = _resolve_device(model_cfg.device)
    dual = preproc_cfg.polarity_invariant
    num_workers = model_cfg.num_workers

    print(f"  Пристрій: {device}")
    print(f"  Модель: {model_cfg.name}")
    print(f"  CPU workers: {num_workers}")
    if preproc_cfg.equalization == "clahe":
        print(f"  Еквалізація: CLAHE (clip={preproc_cfg.clahe_clip_limit}, grid={preproc_cfg.clahe_grid_size})")
    elif preproc_cfg.equalization == "he":
        print(f"  Еквалізація: глобальна гістограмна (HE)")
    else:
        print(f"  Еквалізація: вимкнено")
    if dual:
        print(f"  Polarity invariant (dual-pass): увімкнено")
    if preproc_cfg.l2_normalize:
        print(f"  L2-нормалізація ознак: увімкнено")

    model = _load_model(model_cfg.name, device)
    transform = _build_transform(model_cfg.image_size)

    dataset = _ImageDataset(image_paths, preproc_cfg, transform, dual, model_cfg.image_size)
    loader = DataLoader(
        dataset,
        batch_size=model_cfg.batch_size,
        num_workers=num_workers,
        pin_memory=model_cfg.pin_memory and device.type == "cuda",
        prefetch_factor=model_cfg.prefetch_factor if num_workers > 0 else None,
        persistent_workers=num_workers > 0,
    )

    all_feats_normal: List[np.ndarray] = []
    all_feats_inv: List[np.ndarray] = []

    for batch in tqdm(loader, desc="  Витягування ознак", unit="batch"):
        batch_normal = batch[0].to(device, non_blocking=True)
        with torch.no_grad():
            feats_normal = model(batch_normal)
        all_feats_normal.append(feats_normal.cpu().numpy())

        if dual:
            batch_inv = batch[1].to(device, non_blocking=True)
            with torch.no_grad():
                feats_inv = model(batch_inv)
            all_feats_inv.append(feats_inv.cpu().numpy())

    features = np.concatenate(all_feats_normal, axis=0)
    if dual:
        features_inv = np.concatenate(all_feats_inv, axis=0)
        features = (features + features_inv) / 2.0

    if preproc_cfg.l2_normalize:
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-8)
        features = features / norms

    print(f"  Розмір ознак: {features.shape}")

    _save_cache(cache_cfg, cache_key, features)
    return features
