from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from umap import UMAP

from .config import AccelerationConfig, ReductionConfig

_LAST_REDUCTION_BACKEND = "cpu"


def _resolve_backend(accel_cfg: AccelerationConfig) -> str:
    mode = accel_cfg.backend
    if mode == "cpu":
        return "cpu"
    try:
        import cupy  # noqa: F401
        from cuml.decomposition import PCA as _  # noqa: F401
        from cuml.manifold import UMAP as _  # noqa: F401
        return "rapids"
    except Exception:
        if mode == "cuda":
            print("  [WARN] CUDA backend недоступний, fallback на CPU.")
        return "cpu"


def reduce_dimensions(
    features: np.ndarray,
    cfg: ReductionConfig,
    accel_cfg: AccelerationConfig,
) -> np.ndarray:
    """Two-stage dimensionality reduction: PCA then UMAP.

    Input:  (N, D)  — raw feature vectors from the model
    Output: (N, umap_components) — compact embeddings for clustering
    """
    n_samples, n_features = features.shape
    features = features.astype(np.float32, copy=False)
    global _LAST_REDUCTION_BACKEND
    backend = _resolve_backend(accel_cfg)
    print(f"  Backend (reduction): {'RAPIDS (GPU)' if backend == 'rapids' else 'CPU'}")

    # --- PCA ---
    pca_dim = min(cfg.pca_components, n_features, n_samples)
    print(f"  PCA: {n_features} -> {pca_dim}")
    if backend == "rapids":
        try:
            import cupy as cp
            from cuml.decomposition import PCA as cuPCA

            x_gpu = cp.asarray(features)
            # cuML API can differ across versions; use conservative kwargs first.
            try:
                pca = cuPCA(n_components=pca_dim, output_type="cupy")
            except TypeError:
                pca = cuPCA(n_components=pca_dim)
            reduced_gpu = pca.fit_transform(x_gpu)
            reduced = cp.asnumpy(reduced_gpu)
            evr = getattr(pca, "explained_variance_ratio_", None)
            if evr is not None:
                explained = float(cp.asnumpy(evr).sum() * 100)
            else:
                explained = float("nan")
        except Exception as exc:
            print(
                f"  [WARN] CUDA PCA error ({type(exc).__name__}: {exc}), fallback на CPU."
            )
            backend = "cpu"

    if backend == "cpu":
        pca = PCA(n_components=pca_dim, random_state=42)
        reduced = pca.fit_transform(features)
        explained = pca.explained_variance_ratio_.sum() * 100
    print(f"  PCA пояснює {explained:.1f}% дисперсії")

    # --- UMAP ---
    umap_dim = min(cfg.umap_components, pca_dim)
    n_neighbors = min(cfg.umap_n_neighbors, n_samples - 1)
    print(f"  UMAP: {pca_dim} -> {umap_dim} (n_neighbors={n_neighbors}, metric={cfg.umap_metric})")

    if backend == "rapids":
        try:
            import cupy as cp
            from cuml.manifold import UMAP as cuUMAP

            reduced_gpu = cp.asarray(reduced)
            try:
                umap = cuUMAP(
                    n_components=umap_dim,
                    n_neighbors=n_neighbors,
                    min_dist=cfg.umap_min_dist,
                    metric=cfg.umap_metric,
                    random_state=42,
                    verbose=False,
                    output_type="cupy",
                )
            except TypeError:
                umap = cuUMAP(
                    n_components=umap_dim,
                    n_neighbors=n_neighbors,
                    min_dist=cfg.umap_min_dist,
                    metric=cfg.umap_metric,
                    random_state=42,
                    verbose=False,
                )
            embedded_gpu = umap.fit_transform(reduced_gpu)
            embedded = cp.asnumpy(embedded_gpu)
        except Exception as exc:
            print(
                f"  [WARN] CUDA UMAP error ({type(exc).__name__}: {exc}), fallback на CPU."
            )
            umap = UMAP(
                n_components=umap_dim,
                n_neighbors=n_neighbors,
                min_dist=cfg.umap_min_dist,
                metric=cfg.umap_metric,
                random_state=42,
                verbose=False,
            )
            embedded = umap.fit_transform(reduced)
    else:
        umap = UMAP(
            n_components=umap_dim,
            n_neighbors=n_neighbors,
            min_dist=cfg.umap_min_dist,
            metric=cfg.umap_metric,
            random_state=42,
            verbose=False,
        )
        embedded = umap.fit_transform(reduced)
    print(f"  Фінальна розмірність: {embedded.shape}")
    _LAST_REDUCTION_BACKEND = backend

    return embedded
