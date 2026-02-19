from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import yaml


DEFAULTS = {
    "input_dir": "",
    "output_dir": "",
    "mode": "copy",
    "image_extensions": [".jpg", ".jpeg", ".png", ".bmp", ".tiff"],
    "model": {
        "name": "dinov2_vits14",
        "batch_size": 32,
        "image_size": 224,
        "device": "auto",
        "num_workers": 4,
        "prefetch_factor": 2,
        "pin_memory": True,
    },
    "preprocessing": {
        "equalization": "he",
        "clahe_clip_limit": 3.0,
        "clahe_grid_size": 8,
        "polarity_invariant": True,
        "l2_normalize": True,
        "trim_top": 0.0,
        "trim_bottom": 0.0,
        "trim_left": 0.0,
        "trim_right": 0.0,
    },
    "reduction": {
        "pca_components": 50,
        "umap_components": 10,
        "umap_n_neighbors": 15,
        "umap_min_dist": 0.1,
        "umap_metric": "cosine",
    },
    "clustering": {
        "min_cluster_size": 50,
        "min_samples": 10,
        "cluster_selection_epsilon": 0.0,
        "cluster_selection_method": "leaf",
        "noise_handling": "nearest",
    },
    "visualization": {
        "enabled": True,
        "viz_dir": "viz",
        "max_points": 5000,
        "thumbnail_size": 80,
        "samples_per_cluster": 16,
        "clahe_comparison_samples": 8,
    },
    "annotations": {
        "enabled": True,
        "image_subdir": "img",
        "label_subdir": "lab",
        "label_extension": ".txt",
    },
    "cache": {
        "enabled": True,
        "cache_dir": ".cache",
    },
    "acceleration": {
        "backend": "auto",
    },
    "purity": {
        "enabled": True,
        "fail_on_fail": True,
        "max_centroid_cosine": 0.92,
        "min_silhouette": 0.25,
        "silhouette_sample_size": 10000,
        "silhouette_random_state": 42,
        "max_nn_cross_ratio": 0.5,
        "k_neighbors": 5,
    },
}


@dataclass
class ModelConfig:
    name: str = "dinov2_vits14"
    batch_size: int = 32
    image_size: int = 224
    device: str = "auto"
    num_workers: int = 4
    prefetch_factor: int = 2
    pin_memory: bool = True


@dataclass
class PreprocessingConfig:
    equalization: str = "he"
    clahe_clip_limit: float = 3.0
    clahe_grid_size: int = 8
    polarity_invariant: bool = True
    l2_normalize: bool = True
    trim_top: float = 0.0
    trim_bottom: float = 0.0
    trim_left: float = 0.0
    trim_right: float = 0.0


@dataclass
class ReductionConfig:
    pca_components: int = 50
    umap_components: int = 10
    umap_n_neighbors: int = 15
    umap_min_dist: float = 0.1
    umap_metric: str = "cosine"


@dataclass
class ClusteringConfig:
    min_cluster_size: int = 50
    min_samples: int = 10
    cluster_selection_epsilon: float = 0.0
    cluster_selection_method: str = "leaf"
    noise_handling: str = "nearest"


@dataclass
class VisualizationConfig:
    enabled: bool = True
    viz_dir: str = "viz"
    max_points: int = 5000
    thumbnail_size: int = 80
    samples_per_cluster: int = 16
    clahe_comparison_samples: int = 8


@dataclass
class AnnotationsConfig:
    enabled: bool = True
    image_subdir: str = "img"
    label_subdir: str = "lab"
    label_extension: str = ".txt"


@dataclass
class CacheConfig:
    enabled: bool = True
    cache_dir: str = ".cache"


@dataclass
class AccelerationConfig:
    backend: str = "auto"


@dataclass
class PurityConfig:
    enabled: bool = True
    fail_on_fail: bool = True
    max_centroid_cosine: float = 0.92
    min_silhouette: float = 0.25
    silhouette_sample_size: int = 10000
    silhouette_random_state: int = 42
    max_nn_cross_ratio: float = 0.5
    k_neighbors: int = 5


@dataclass
class Config:
    input_dir: Path = field(default_factory=lambda: Path(""))
    output_dir: Path = field(default_factory=lambda: Path(""))
    mode: str = "copy"
    image_extensions: List[str] = field(
        default_factory=lambda: [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
    )
    model: ModelConfig = field(default_factory=ModelConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    reduction: ReductionConfig = field(default_factory=ReductionConfig)
    clustering: ClusteringConfig = field(default_factory=ClusteringConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    annotations: AnnotationsConfig = field(default_factory=AnnotationsConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    acceleration: AccelerationConfig = field(default_factory=AccelerationConfig)
    purity: PurityConfig = field(default_factory=PurityConfig)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (non-destructive)."""
    merged = base.copy()
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _validate(cfg: Config) -> None:
    if not cfg.input_dir or not cfg.input_dir.exists():
        print(f"[ПОМИЛКА] input_dir не існує: {cfg.input_dir}")
        sys.exit(1)

    if not cfg.output_dir:
        print("[ПОМИЛКА] output_dir не задано")
        sys.exit(1)

    if cfg.mode not in ("copy", "move"):
        print(f"[ПОМИЛКА] mode має бути 'copy' або 'move', отримано: {cfg.mode}")
        sys.exit(1)

    valid_models = ("dinov2_vits14", "dinov2_vitb14", "resnet50")
    if cfg.model.name not in valid_models:
        print(f"[ПОМИЛКА] model.name має бути одним із {valid_models}, отримано: {cfg.model.name}")
        sys.exit(1)

    if cfg.model.device not in ("auto", "cuda", "cpu"):
        print(f"[ПОМИЛКА] model.device має бути 'auto', 'cuda' або 'cpu', отримано: {cfg.model.device}")
        sys.exit(1)

    valid_methods = ("eom", "leaf")
    if cfg.clustering.cluster_selection_method not in valid_methods:
        print(
            f"[ПОМИЛКА] clustering.cluster_selection_method має бути одним із {valid_methods}, "
            f"отримано: {cfg.clustering.cluster_selection_method}"
        )
        sys.exit(1)

    if cfg.clustering.noise_handling not in ("separate", "nearest"):
        print(
            f"[ПОМИЛКА] clustering.noise_handling має бути 'separate' або 'nearest', "
            f"отримано: {cfg.clustering.noise_handling}"
        )
        sys.exit(1)

    valid_metrics = ("cosine", "euclidean")
    if cfg.reduction.umap_metric not in valid_metrics:
        print(
            f"[ПОМИЛКА] reduction.umap_metric має бути одним із {valid_metrics}, "
            f"отримано: {cfg.reduction.umap_metric}"
        )
        sys.exit(1)

    if cfg.purity.k_neighbors < 1:
        print("[ПОМИЛКА] purity.k_neighbors має бути >= 1")
        sys.exit(1)

    if cfg.purity.silhouette_sample_size < 1:
        print("[ПОМИЛКА] purity.silhouette_sample_size має бути >= 1")
        sys.exit(1)

    if cfg.acceleration.backend not in ("auto", "cpu", "cuda"):
        print("[ПОМИЛКА] acceleration.backend має бути 'auto', 'cpu' або 'cuda'")
        sys.exit(1)


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path)
    if not path.exists():
        print(f"[ПОМИЛКА] Файл конфігурації не знайдено: {path}")
        sys.exit(1)

    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    merged = _deep_merge(DEFAULTS, raw)

    cfg = Config(
        input_dir=Path(merged["input_dir"]),
        output_dir=Path(merged["output_dir"]),
        mode=merged["mode"],
        image_extensions=[ext.lower() for ext in merged["image_extensions"]],
        model=ModelConfig(**merged["model"]),
        preprocessing=PreprocessingConfig(**merged["preprocessing"]),
        reduction=ReductionConfig(**merged["reduction"]),
        clustering=ClusteringConfig(**merged["clustering"]),
        visualization=VisualizationConfig(**merged["visualization"]),
        annotations=AnnotationsConfig(**merged["annotations"]),
        cache=CacheConfig(**merged["cache"]),
        acceleration=AccelerationConfig(**merged["acceleration"]),
        purity=PurityConfig(**merged["purity"]),
    )

    _validate(cfg)
    return cfg
